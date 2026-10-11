"""
Cluster-wide KV Cache Block Memory Manager.
Coordinates block allocation across all tensor parallel shards to ensure synchronized KV caches.
"""

from typing import Dict, List, Optional
import math
from engine.config import KVCacheConfig
from engine.scheduler.sequence import Sequence, SequenceStatus


class ClusterMemoryManager:
    """
    Centralized block allocator running on the coordinator node.
    Synchronizes physical block table mappings across all worker shards.
    """

    def __init__(self, kv_config: KVCacheConfig):
        self.block_size = kv_config.block_size
        self.total_blocks = kv_config.num_blocks
        self.free_blocks: List[int] = list(range(self.total_blocks))
        self.seq_block_tables: Dict[str, List[int]] = {}

    @property
    def free_block_count(self) -> int:
        return len(self.free_blocks)

    @property
    def used_block_count(self) -> int:
        return self.total_blocks - len(self.free_blocks)

    @property
    def memory_utilization(self) -> float:
        return (self.used_block_count / self.total_blocks) if self.total_blocks > 0 else 0.0

    def can_allocate_prefill(self, sequence: Sequence) -> bool:
        """Checks if enough free blocks exist to admit this sequence's prompt."""
        needed = math.ceil(len(sequence.prompt_token_ids) / self.block_size)
        return len(self.free_blocks) >= needed

    def allocate_prefill(self, sequence: Sequence) -> List[int]:
        """Allocates initial blocks for prompt prefill."""
        needed = math.ceil(len(sequence.prompt_token_ids) / self.block_size)
        if len(self.free_blocks) < needed:
            raise MemoryError(f"Insufficient KV cache blocks: needed {needed}, available {len(self.free_blocks)}")

        allocated = []
        for _ in range(needed):
            allocated.append(self.free_blocks.pop(0))

        self.seq_block_tables[sequence.seq_id] = allocated
        return allocated

    def can_append_token(self, sequence: Sequence) -> bool:
        """Checks if a new block is required and available to append one token."""
        curr_len = sequence.total_length
        if curr_len % self.block_size == 0:
            # Need a new block
            return len(self.free_blocks) > 0
        return True

    def append_token_slot(self, sequence: Sequence) -> Optional[int]:
        """
        Extends block allocation if current block is full.
        Returns newly allocated block_id if a new block was added, else None.
        """
        curr_len = sequence.total_length
        if curr_len % self.block_size == 0:
            if not self.free_blocks:
                raise MemoryError(f"KV cache full: cannot extend sequence {sequence.seq_id}")
            new_block = self.free_blocks.pop(0)
            self.seq_block_tables[sequence.seq_id].append(new_block)
            return new_block
        return None

    def free_sequence(self, sequence: Sequence) -> List[int]:
        """Reclaims all blocks allocated to a sequence."""
        blocks = self.seq_block_tables.pop(sequence.seq_id, [])
        self.free_blocks.extend(blocks)
        return blocks

    def get_block_table(self, seq_id: str) -> List[int]:
        return self.seq_block_tables.get(seq_id, [])
