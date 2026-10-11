"""
Continuous Batching Scheduler (Iteration-Level Scheduling).
Dynamically schedules in-flight sequence decodes and interleaves new sequence prefills
to maximize token generation throughput and hardware utilization.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time

from engine.config import SchedulerConfig, KVCacheConfig
from engine.scheduler.sequence import Sequence, SequenceStatus
from engine.scheduler.memory_manager import ClusterMemoryManager


@dataclass
class BatchSchedule:
    """Represents the workload for a single iteration step across the cluster."""
    step_id: int
    prefill_sequences: List[Sequence]
    decode_sequences: List[Sequence]
    block_tables: Dict[str, List[int]]

    @property
    def total_sequences(self) -> int:
        return len(self.prefill_sequences) + len(self.decode_sequences)

    @property
    def is_empty(self) -> bool:
        return self.total_sequences == 0


class ContinuousBatcher:
    """
    Core iteration-level scheduler.
    Implements continuous batching by forming dynamic batches every forward pass step.
    """

    def __init__(self, scheduler_config: SchedulerConfig, kv_config: KVCacheConfig):
        self.config = scheduler_config
        self.memory_manager = ClusterMemoryManager(kv_config)
        self.waiting_queue: List[Sequence] = []
        self.running_sequences: List[Sequence] = []
        self.finished_sequences: List[Sequence] = []
        self.step_counter: int = 0
        self.total_tokens_generated: int = 0

    def add_request(self, sequence: Sequence):
        """Adds a new client sequence to the admission queue."""
        sequence.status = SequenceStatus.WAITING
        self.waiting_queue.append(sequence)

    def schedule_step(self) -> BatchSchedule:
        """
        Determines the batch of sequences to run in the next forward execution step.
        Prioritizes existing running sequences (decode phase) to minimize latency,
        then admits waiting sequences (prefill phase) up to max_batch_size and memory capacity.
        """
        self.step_counter += 1
        active_decodes: List[Sequence] = []
        preempted_sequences: List[Sequence] = []

        # 1. Process RUNNING sequences (Decode Phase)
        for seq in self.running_sequences:
            if self.memory_manager.can_append_token(seq):
                self.memory_manager.append_token_slot(seq)
                active_decodes.append(seq)
            else:
                # KV cache memory pressure: preempt sequence
                seq.status = SequenceStatus.PREEMPTED
                self.memory_manager.free_sequence(seq)
                preempted_sequences.append(seq)

        # Preempted sequences return to waiting queue
        if preempted_sequences:
            self.waiting_queue = preempted_sequences + self.waiting_queue

        # 2. Process WAITING sequences (Prefill Phase)
        active_prefills: List[Sequence] = []
        remaining_capacity = self.config.max_batch_size - len(active_decodes)

        admitted_indices = []
        for i, seq in enumerate(self.waiting_queue):
            if len(active_prefills) >= remaining_capacity:
                break
            if self.memory_manager.can_allocate_prefill(seq):
                self.memory_manager.allocate_prefill(seq)
                seq.status = SequenceStatus.RUNNING
                active_prefills.append(seq)
                admitted_indices.append(i)

        # Remove admitted from waiting queue
        for idx in reversed(admitted_indices):
            self.waiting_queue.pop(idx)

        # Update running set
        self.running_sequences = active_decodes + active_prefills

        # Gather block tables
        block_tables = {}
        for seq in self.running_sequences:
            block_tables[seq.seq_id] = self.memory_manager.get_block_table(seq.seq_id)

        return BatchSchedule(
            step_id=self.step_counter,
            prefill_sequences=active_prefills,
            decode_sequences=active_decodes,
            block_tables=block_tables,
        )

    def process_step_outputs(
        self, schedule: BatchSchedule, sampled_tokens: Dict[str, int]
    ) -> List[Tuple[Sequence, int, bool]]:
        """
        Processes newly generated tokens, advances sequences, and frees memory for finished requests.
        Returns: List of tuples (sequence, sampled_token_id, is_finished).
        """
        results = []
        still_running = []

        all_step_seqs = schedule.prefill_sequences + schedule.decode_sequences

        for seq in all_step_seqs:
            token_id = sampled_tokens.get(seq.seq_id)
            if token_id is None:
                continue

            seq.append_token(token_id)
            self.total_tokens_generated += 1

            if seq.is_finished:
                self.memory_manager.free_sequence(seq)
                self.finished_sequences.append(seq)
                results.append((seq, token_id, True))
            else:
                still_running.append(seq)
                results.append((seq, token_id, False))

        self.running_sequences = still_running
        return results
