"""
Distributed Sharded LLaMA / Mistral Transformer Architecture.
Assembles tensor-parallel Attention, SwiGLU MLP, RMSNorm, and Paged KV Cache.
"""

from typing import Dict, List, Optional, Tuple, Union
import math

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object

from engine.config import ModelConfig
from engine.tensor_parallel.layers import RMSNorm
from engine.tensor_parallel.comm import DistributedCommunicator
from engine.model.attention import DistributedAttention
from engine.model.mlp import DistributedMLP
from engine.model.kv_cache import PhysicalKVCachePool


class DistributedTransformerBlock(nn.Module if HAS_TORCH else object):
    """A single Transformer layer with tensor-parallel attention and MLP."""

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
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.self_attn = DistributedAttention(config, layer_idx, rank, world_size, communicator)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.mlp = DistributedMLP(config, layer_idx, rank, world_size, communicator)

    def set_communicator(self, communicator: DistributedCommunicator):
        self.self_attn.set_communicator(communicator)
        self.mlp.set_communicator(communicator)

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
        Residual connection + Attention + Residual connection + MLP
        """
        # Attention block with pre-norm
        norm_x = self.input_layernorm(x)
        attn_out = self.self_attn(
            norm_x,
            positions=positions,
            kv_pool=kv_pool,
            block_tables=block_tables,
            is_prefill=is_prefill,
            step_id=step_id,
        )
        h = x + attn_out

        # MLP block with pre-norm
        norm_h = self.post_attention_layernorm(h)
        mlp_out = self.mlp(norm_h, step_id=step_id)
        out = h + mlp_out
        return out


class DistributedLLaMAModel(nn.Module if HAS_TORCH else object):
    """
    Complete Distributed LLaMA Architecture.
    Can be run as a full monolithic model (world_size=1) or as an individual shard (rank < world_size).
    """

    def __init__(
        self,
        config: ModelConfig,
        rank: int = 0,
        world_size: int = 1,
        communicator: Optional[DistributedCommunicator] = None,
        device: str = "cpu",
    ):
        if HAS_TORCH:
            super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.communicator = communicator
        self.device = device

        if HAS_TORCH:
            # Word embeddings
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

            # Transformer layers
            self.layers = nn.ModuleList(
                [
                    DistributedTransformerBlock(
                        config=config,
                        layer_idx=i,
                        rank=rank,
                        world_size=world_size,
                        communicator=communicator,
                    )
                    for i in range(config.num_layers)
                ]
            )

            # Final layer norm and LM head
            self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

            self.to(device)

    def set_communicator(self, communicator: DistributedCommunicator):
        self.communicator = communicator
        if HAS_TORCH:
            for layer in self.layers:
                layer.set_communicator(communicator)

    def forward(
        self,
        token_ids: "torch.Tensor",
        positions: "torch.Tensor",
        kv_pool: Optional[PhysicalKVCachePool] = None,
        block_tables: Optional[List[List[int]]] = None,
        is_prefill: bool = True,
        step_id: int = 0,
        return_logits_last_only: bool = True,
    ) -> "torch.Tensor":
        """
        token_ids: [batch, seq_len]
        positions: [batch, seq_len]
        returns: logits [batch, vocab_size] (if last_only) or [batch, seq_len, vocab_size]
        """
        if not HAS_TORCH:
            return None

        # 1. Embedding lookup
        h = self.embed_tokens(token_ids)

        # 2. Sequential transformer layer blocks
        for layer in self.layers:
            h = layer(
                h,
                positions=positions,
                kv_pool=kv_pool,
                block_tables=block_tables,
                is_prefill=is_prefill,
                step_id=step_id,
            )

        # 3. Final normalization
        h = self.norm(h)

        # 4. LM Head projection
        if return_logits_last_only:
            # Extract last token hidden state for each sequence in batch
            h_last = h[:, -1:, :]
            logits = self.lm_head(h_last).squeeze(1)  # [batch, vocab_size]
        else:
            logits = self.lm_head(h)  # [batch, seq_len, vocab_size]

        return logits
