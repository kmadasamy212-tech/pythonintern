"""
Unit tests for distributed gRPC collective communications and All-Reduce.
"""

import threading
import pytest
import torch
from engine.tensor_parallel.comm import DistributedCommunicator


def test_all_reduce_4_ranks():
    """
    Simulates a 4-node cluster performing All-Reduce SUM.
    Each node holds tensor T_r = (r + 1) * ones.
    Expected result on every node: sum_{r=0}^3 (r + 1) * ones = (1 + 2 + 3 + 4) * ones = 10 * ones.
    """
    world_size = 4
    shape = (2, 8)
    comms = [DistributedCommunicator(rank=r, world_size=world_size) for r in range(world_size)]

    # Interconnect all ranks
    for i in range(world_size):
        for j in range(world_size):
            if i != j:
                comms[i].register_in_memory_peer(j, comms[j])

    results = [None] * world_size

    def worker_thread(rank: int):
        input_tensor = torch.full(shape, float(rank + 1), dtype=torch.float32)
        res = comms[rank].all_reduce_sum(input_tensor, step_tag="step_test_collectives")
        results[rank] = res

    threads = [threading.Thread(target=worker_thread, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = torch.full(shape, 10.0, dtype=torch.float32)
    for r in range(world_size):
        assert results[r] is not None
        assert torch.allclose(results[r], expected)
