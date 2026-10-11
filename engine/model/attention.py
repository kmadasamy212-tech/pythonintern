"""
Sharded Multi-Head / Grouped-Query Attention (GQA) with RoPE and Paged KV Cache.
Runs tensor-parallel attention with local head sharding and gRPC All-Reduce out-projection.
"""

import math
from typing import Optional, Tuple, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object

from engine.config import ModelConfig
from engine.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from engine.tensor_parallel.comm import DistributedCommunicator
from engine.model.kv_cache import PhysicalKVCachePool


class RotaryEmbedding:
    """Rotary Positional Embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0, device: str = "cpu"):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.device = device
        self._init_cos_sin_tables()

    def _init_cos_sin_tables(self):
        if not HAS_TORCH:
            return
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=self.device) / self.dim))
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.device)
        freqs = torch.outer(t, inv_freq)
        # Shape: [max_seq_len, dim // 2, 2]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def apply_rope(self, x: "torch.Tensor", positions: "torch.Tensor") -> "torch.Tensor":
        """
        x: [batch, seq_len, num_heads, head_dim] or [seq_len, num_heads, head_dim]
        positions: [batch, seq_len] or [seq_len]
        """
        if not HAS_TORCH:
            return x

        # Gather cos and sin for positions
        cos = self.cos_cached[positions].unsqueeze(-2)  # [batch, seq_len, 1, head_dim]
        sin = self.sin_cached[positions].unsqueeze(-2)

        # Rotate half: [-x2, x1]
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        x_rotated = torch.cat((-x2, x1), dim=-1)

        return (x * cos) + (x_rotated * sin)


def repeat_kv(x: "torch.Tensor", num_rep: int) -> "torch.Tensor":
    """
    Expands KV heads for Grouped Query Attention (GQA).
    x: [batch, seq_len, num_kv_heads, head_dim]
    returns: [batch, seq_len, num_kv_heads * num_rep, head_dim]
    """
    if num_rep == 1:
        return x
    batch, seq_len, num_kv_heads, head_dim = x.shape
    x = x[:, :, :, None, :].expand(batch, seq_len, num_kv_heads, num_rep, head_dim)
    return x.reshape(batch, seq_len, num_kv_heads * num_rep, head_dim)


class DistributedAttention(nn.Module if HAS_TORCH else object):
    """
    Tensor Parallel Attention block:
    - Q, K, V are sharded column-parallel (each worker computes num_heads // world_size)
    - Keys and Values are stored in local Paged KV Cache partition
    - Out-projection is sharded row-parallel with gRPC All-Reduce
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        rank: int = 0,
        world_size: int = 1,
        communicator: Optional[DistributedCommunicator] = None,
    ):
        if HAS_TORCH:
            super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.rank = rank
        self.world_size = world_size
        self.head_dim = config.head_dim
        self.communicator = communicator

        assert config.num_heads % world_size == 0
        assert config.num_kv_heads % world_size == 0
        self.local_heads = config.num_heads // world_size
        self.local_kv_heads = config.num_kv_heads // world_size
        self.num_queries_per_kv = self.local_heads // self.local_kv_heads

        # Projections
        self.q_proj = ColumnParallelLinear(
            config.hidden_dim, config.num_heads * self.head_dim, bias=False, rank=rank, world_size=world_size
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False, rank=rank, world_size=world_size
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False, rank=rank, world_size=world_size
        )
        self.out_proj = RowParallelLinear(
            config.num_heads * self.head_dim,
            config.hidden_dim,
            bias=False,
            rank=rank,
            world_size=world_size,
            communicator=communicator,
        )

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.max_seq_len)

    def set_communicator(self, communicator: DistributedCommunicator):
        self.communicator = communicator
        self.out_proj.set_communicator(communicator)

    def forward(
        self,
        x: "torch.Tensor",
        positions: "torch.Tensor",
        kv_pool: Optional[PhysicalKVCachePool] = None,
        block_tables: Optional[List[List[int]]] = None,
        is_prefill: bool = True,
        step_id: int = 0,
    ) -> "torch.Tensor":
        """
        x: [batch, seq_len, hidden_dim]
        positions: [batch, seq_len]
        """
        if not HAS_TORCH:
            return x

        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V on local heads
        # Q: [batch, seq_len, local_heads * head_dim]
        # K, V: [batch, seq_len, local_kv_heads * head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.local_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.local_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.local_kv_heads, self.head_dim)

        # Apply RoPE
        q = self.rope.apply_rope(q, positions)
        k = self.rope.apply_rope(k, positions)

        # Handle KV Cache updates
        if kv_pool is not None and block_tables is not None:
            if is_prefill:
                # Store prefill prompt tokens into cache blocks
                for b_idx in range(batch_size):
                    b_table = block_tables[b_idx]
                    for t_pos in range(seq_len):
                        blk_id = b_table[t_pos // kv_pool.block_size]
                        off = t_pos % kv_pool.block_size
                        kv_pool.store_kv(self.layer_idx, blk_id, off, k[b_idx, t_pos], v[b_idx, t_pos])
                k_full, v_full = k, v
            else:
                # Single-token decode step: seq_len == 1
                curr_k_list, curr_v_list = [], []
                for b_idx in range(batch_size):
                    b_table = block_tables[b_idx]
                    curr_len = int(positions[b_idx, 0].item()) + 1
                    # Store current new token
                    blk_id = b_table[(curr_len - 1) // kv_pool.block_size]
                    off = (curr_len - 1) % kv_pool.block_size
                    kv_pool.store_kv(self.layer_idx, blk_id, off, k[b_idx, 0], v[b_idx, 0])
                    # Read full sequence KV from physical pool
                    k_seq, v_seq = kv_pool.read_kv_sequence(self.layer_idx, b_table, curr_len)
                    curr_k_list.append(k_seq)
                    curr_v_list.append(v_seq)

                # Pad to max len in current decode batch if lengths differ
                max_len = max(k_s.shape[0] for k_s in curr_k_list)
                k_padded = torch.zeros(
                    batch_size, max_len, self.local_kv_heads, self.head_dim, dtype=q.dtype, device=x.device
                )
                v_padded = torch.zeros(
                    batch_size, max_len, self.local_kv_heads, self.head_dim, dtype=q.dtype, device=x.device
                )
                for b_idx in range(batch_size):
                    l = curr_k_list[b_idx].shape[0]
                    k_padded[b_idx, :l] = curr_k_list[b_idx]
                    v_padded[b_idx, :l] = curr_v_list[b_idx]

                k_full, v_full = k_padded, v_padded
        else:
            k_full, v_full = k, v

        # Repeat KV heads for GQA if needed
        k_full = repeat_kv(k_full, self.num_queries_per_kv)
        v_full = repeat_kv(v_full, self.num_queries_per_kv)

        # Transpose for scaled dot product: [batch, num_heads, seq_len, head_dim]
        q_t = q.transpose(1, 2)
        k_t = k_full.transpose(1, 2)
        v_t = v_full.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

        if is_prefill:
            # Causal mask for prefill
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
            scores = scores + causal_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v_t)  # [batch, local_heads, seq_len, head_dim]

        # Reshape back to [batch, seq_len, local_heads * head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Row-parallel Out Projection + All-Reduce
        step_tag = f"step_{step_id}_l{self.layer_idx}_attn"
        output = self.out_proj(attn_out, step_tag=step_tag)
        return output
