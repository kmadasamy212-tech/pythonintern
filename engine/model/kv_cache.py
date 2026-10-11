"""
Paged KV Cache and Block Allocator.
Stores Key-Value states in discrete physical blocks to eliminate internal fragmentation
and support iteration-level continuous batching on commodity hardware.
"""

from typing import Dict, List, Optional, Tuple
import math

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from engine.config import KVCacheConfig, ModelConfig


class BlockTable:
    """Logical mapping of token indices to physical block indices for one sequence."""

    def __init__(self, seq_id: str, block_size: int):
        self.seq_id = seq_id
        self.block_size = block_size
        self.physical_block_ids: List[int] = []
        self.num_tokens: int = 0

    def append_block(self, block_id: int):
        self.physical_block_ids.append(block_id)

    def num_allocated_blocks(self) -> int:
        return len(self.physical_block_ids)

    def get_physical_slot(self, logical_pos: int) -> Tuple[int, int]:
        """Returns (physical_block_id, offset_in_block)."""
        block_idx = logical_pos // self.block_size
        offset = logical_pos % self.block_size
        assert block_idx < len(self.physical_block_ids), (
            f"Logical position {logical_pos} exceeds allocated blocks ({len(self.physical_block_ids)})"
        )
        return self.physical_block_ids[block_idx], offset


class PhysicalKVCachePool:
    """
    Physical tensor pool for Key and Value cache for local shard's KV heads.
    Shape per block: [block_size, num_local_kv_heads, head_dim]
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_local_kv_heads: int,
        head_dim: int,
        num_layers: int,
        device: str = "cpu",
        dtype=None,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_local_kv_heads = num_local_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.device = device

        if HAS_TORCH:
            tensor_dtype = dtype or torch.float32
            # Dimensions: [num_layers, num_blocks, block_size, num_local_kv_heads, head_dim]
            self.k_cache = torch.zeros(
                (num_layers, num_blocks, block_size, num_local_kv_heads, head_dim),
                dtype=tensor_dtype,
                device=device,
            )
            self.v_cache = torch.zeros(
                (num_layers, num_blocks, block_size, num_local_kv_heads, head_dim),
                dtype=tensor_dtype,
                device=device,
            )
        else:
            self.k_cache = None
            self.v_cache = None

    def store_kv(
        self,
        layer_idx: int,
        physical_block_id: int,
        offset: int,
        k: "torch.Tensor",
        v: "torch.Tensor",
    ):
        """
        Stores single token's K and V vectors into the specified physical block slot.
        k: [num_local_kv_heads, head_dim]
        v: [num_local_kv_heads, head_dim]
        """
        if HAS_TORCH:
            self.k_cache[layer_idx, physical_block_id, offset, :, :] = k
            self.v_cache[layer_idx, physical_block_id, offset, :, :] = v

    def store_kv_batch(
        self,
        layer_idx: int,
        block_ids: List[int],
        offsets: List[int],
        k: "torch.Tensor",
        v: "torch.Tensor",
    ):
        """
        Vectorized store for a batch of tokens.
        k: [batch_tokens, num_local_kv_heads, head_dim]
        v: [batch_tokens, num_local_kv_heads, head_dim]
        """
        if not HAS_TORCH:
            return
        for i, (b_id, off) in enumerate(zip(block_ids, offsets)):
            self.k_cache[layer_idx, b_id, off, :, :] = k[i]
            self.v_cache[layer_idx, b_id, off, :, :] = v[i]

    def read_kv_sequence(
        self, layer_idx: int, block_ids: List[int], seq_len: int
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Gathers Key and Value representations for an entire sequence from its block list.
        Returns:
            k_seq: [seq_len, num_local_kv_heads, head_dim]
            v_seq: [seq_len, num_local_kv_heads, head_dim]
        """
        if not HAS_TORCH:
            return None, None

        if not block_ids:
            return (
                torch.empty((0, self.num_local_kv_heads, self.head_dim), device=self.device),
                torch.empty((0, self.num_local_kv_heads, self.head_dim), device=self.device),
            )

        # Slice all full blocks
        k_blocks = self.k_cache[layer_idx, block_ids]  # [len(block_ids), block_size, kv_heads, head_dim]
        v_blocks = self.v_cache[layer_idx, block_ids]

        # Reshape to continuous sequence
        k_all = k_blocks.view(-1, self.num_local_kv_heads, self.head_dim)
        v_all = v_blocks.view(-1, self.num_local_kv_heads, self.head_dim)

        return k_all[:seq_len], v_all[:seq_len]


class KVCacheManager:
    """
    Manages block allocation, freeing, and per-sequence block tables.
    """

    def __init__(self, config: KVCacheConfig, model_config: ModelConfig, world_size: int):
        self.config = config
        self.block_size = config.block_size
        self.total_blocks = config.num_blocks
        self.free_blocks: List[int] = list(range(config.num_blocks))
        self.block_tables: Dict[str, BlockTable] = {}

        num_local_kv_heads = max(1, model_config.num_kv_heads // world_size)
        self.pool = PhysicalKVCachePool(
            num_blocks=config.num_blocks,
            block_size=config.block_size,
            num_local_kv_heads=num_local_kv_heads,
            head_dim=model_config.head_dim,
            num_layers=model_config.num_layers,
            device=config.device,
        )

    @property
    def free_block_count(self) -> int:
        return len(self.free_blocks)

    @property
    def allocated_block_count(self) -> int:
        return self.total_blocks - len(self.free_blocks)

    def allocate_blocks_for_sequence(self, seq_id: str, num_tokens: int) -> BlockTable:
        """Allocates enough physical blocks to hold num_tokens for seq_id."""
        needed_blocks = math.ceil(num_tokens / self.block_size)
        if needed_blocks > len(self.free_blocks):
            raise MemoryError(
                f"Cannot allocate {needed_blocks} blocks for seq {seq_id}. "
                f"Only {len(self.free_blocks)} free blocks remaining."
            )

        if seq_id not in self.block_tables:
            self.block_tables[seq_id] = BlockTable(seq_id, self.block_size)

        table = self.block_tables[seq_id]
        current_blocks = len(table.physical_block_ids)
        if needed_blocks > current_blocks:
            for _ in range(needed_blocks - current_blocks):
                b_id = self.free_blocks.pop(0)
                table.append_block(b_id)
        table.num_tokens = num_tokens
        return table

    def append_token(self, seq_id: str) -> Tuple[int, int]:
        """
        Advances sequence by 1 token. Allocates a new block if current block is full.
        Returns (physical_block_id, offset_in_block).
        """
        table = self.block_tables[seq_id]
        new_token_pos = table.num_tokens
        block_idx = new_token_pos // self.block_size
        offset = new_token_pos % self.block_size

        if block_idx >= len(table.physical_block_ids):
            if not self.free_blocks:
                raise MemoryError(f"Out of KV cache memory. Cannot allocate block for seq {seq_id}.")
            b_id = self.free_blocks.pop(0)
            table.append_block(b_id)

        table.num_tokens += 1
        return table.physical_block_ids[block_idx], offset

    def free_sequence(self, seq_id: str) -> int:
        """Frees all blocks associated with a sequence."""
        if seq_id not in self.block_tables:
            return 0
        table = self.block_tables.pop(seq_id)
        freed = len(table.physical_block_ids)
        self.free_blocks.extend(table.physical_block_ids)
        return freed
