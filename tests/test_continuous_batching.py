"""
Unit tests for Continuous Batching Scheduler and token generation progression.
"""

import pytest
from engine.config import SchedulerConfig, KVCacheConfig
from engine.scheduler.sequence import Sequence, SequenceStatus
from engine.scheduler.continuous_batcher import ContinuousBatcher


def test_continuous_batching_interleaving():
    """
    Verifies that ContinuousBatcher dynamically interleaves new request prefills
    with active ongoing decodes without stalling.
    """
    scheduler_cfg = SchedulerConfig(max_batch_size=4)
    kv_cfg = KVCacheConfig(block_size=8, num_blocks=20)
    batcher = ContinuousBatcher(scheduler_cfg, kv_cfg)

    # 1. Enqueue Req 1 (short prompt) and Req 2 (longer prompt)
    seq1 = Sequence("seq1", prompt="hello", prompt_token_ids=[1, 2, 3], max_new_tokens=4)
    seq2 = Sequence("seq2", prompt="world", prompt_token_ids=[4, 5], max_new_tokens=5)
    batcher.add_request(seq1)
    batcher.add_request(seq2)

    # Step 1: Both are admitted as Prefills
    s1 = batcher.schedule_step()
    assert len(s1.prefill_sequences) == 2
    assert len(s1.decode_sequences) == 0

    # Process step 1 output: both generate 1 token
    results1 = batcher.process_step_outputs(s1, {"seq1": 101, "seq2": 102})
    assert len(results1) == 2
    assert seq1.status == SequenceStatus.RUNNING
    assert seq2.status == SequenceStatus.RUNNING
    assert seq1.output_token_ids == [101]

    # While seq1 and seq2 are decoding, a new request (seq3) arrives!
    seq3 = Sequence("seq3", prompt="new arrival", prompt_token_ids=[7, 8, 9], max_new_tokens=3)
    batcher.add_request(seq3)

    # Step 2: seq1 and seq2 are Decodes, seq3 is admitted as Prefill in the SAME batch!
    s2 = batcher.schedule_step()
    assert len(s2.decode_sequences) == 2
    assert len(s2.prefill_sequences) == 1
    assert s2.prefill_sequences[0].seq_id == "seq3"

    # Process step 2 outputs
    results2 = batcher.process_step_outputs(s2, {"seq1": 103, "seq2": 104, "seq3": 105})
    assert len(results2) == 3
    assert seq1.output_token_ids == [101, 103]
    assert seq3.output_token_ids == [105]


def test_sequence_completion_and_memory_reclaim():
    """
    Verifies that finished requests are evicted and their KV cache memory is returned to the pool.
    """
    scheduler_cfg = SchedulerConfig(max_batch_size=2)
    kv_cfg = KVCacheConfig(block_size=8, num_blocks=10)
    batcher = ContinuousBatcher(scheduler_cfg, kv_cfg)

    initial_free = batcher.memory_manager.free_block_count

    # Max tokens = 2
    seq = Sequence("seq_done", prompt="test", prompt_token_ids=[10, 20], max_new_tokens=2)
    batcher.add_request(seq)

    # Step 1: Prefill
    s1 = batcher.schedule_step()
    used_blocks = batcher.memory_manager.used_block_count
    assert used_blocks > 0

    batcher.process_step_outputs(s1, {"seq_done": 30})
    assert not seq.is_finished

    # Step 2: Decode -> reaches max_new_tokens=2 -> finishes
    s2 = batcher.schedule_step()
    batcher.process_step_outputs(s2, {"seq_done": 40})

    assert seq.is_finished
    assert seq.status == SequenceStatus.FINISHED_LENGTH
    # Memory must be fully reclaimed back to initial pool
    assert batcher.memory_manager.free_block_count == initial_free
