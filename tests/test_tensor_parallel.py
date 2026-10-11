"""
Unit tests for Tensor Parallelism layers, collectives, and weight sharding.
"""

import pytest
import torch
import torch.nn as nn
from engine.config import ModelConfig
from engine.tensor_parallel.buffer_codec import serialize_tensor, deserialize_tensor
from engine.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, RMSNorm
from engine.tensor_parallel.sharding import WeightSharder
from engine.tensor_parallel.comm import DistributedCommunicator


def test_buffer_codec_roundtrip():
    """Verifies zero-overhead serialization and exact numeric reconstruction."""
    t_f32 = torch.randn(4, 16, 32, dtype=torch.float32)
    raw_bytes, shape, dtype_str = serialize_tensor(t_f32)
    reconstructed = deserialize_tensor(raw_bytes, shape, dtype_str)
    assert torch.allclose(t_f32, reconstructed)

    t_i64 = torch.randint(0, 1000, (8, 64), dtype=torch.int64)
    raw_b, shape_b, dtype_b = serialize_tensor(t_i64)
    rec_i64 = deserialize_tensor(raw_b, shape_b, dtype_b)
    assert torch.equal(t_i64, rec_i64)


def test_column_parallel_linear():
    """
    Verifies that ColumnParallelLinear partitions out_features across ranks
    and concatenating rank outputs equals monolithic Linear output.
    """
    batch_size = 2
    seq_len = 4
    in_features = 32
    out_features = 64
    world_size = 2

    # Monolithic reference layer
    full_linear = nn.Linear(in_features, out_features, bias=False)
    x = torch.randn(batch_size, seq_len, in_features)
    expected_full_out = full_linear(x)

    # Shard weights to 2 ranks
    chunk_size = out_features // world_size
    col_rank0 = ColumnParallelLinear(in_features, out_features, bias=False, rank=0, world_size=world_size)
    col_rank1 = ColumnParallelLinear(in_features, out_features, bias=False, rank=1, world_size=world_size)

    with torch.no_grad():
        col_rank0.weight.copy_(full_linear.weight.t()[:, 0:chunk_size])
        col_rank1.weight.copy_(full_linear.weight.t()[:, chunk_size:out_features])

        out0 = col_rank0(x)
        out1 = col_rank1(x)

        combined_out = torch.cat([out0, out1], dim=-1)

    assert torch.allclose(expected_full_out, combined_out, atol=1e-5)


def test_row_parallel_linear_with_comm():
    """
    Verifies that RowParallelLinear partitions in_features across ranks
    and All-Reduce (SUM) of partial outputs equals monolithic Linear output.
    """
    batch_size = 2
    seq_len = 4
    in_features = 32
    out_features = 16
    world_size = 2

    full_linear = nn.Linear(in_features, out_features, bias=False)
    x = torch.randn(batch_size, seq_len, in_features)
    expected_full_out = full_linear(x)

    # In-memory communicators
    comm0 = DistributedCommunicator(rank=0, world_size=world_size)
    comm1 = DistributedCommunicator(rank=1, world_size=world_size)
    comm0.register_in_memory_peer(1, comm1)
    comm1.register_in_memory_peer(0, comm0)

    chunk_size = in_features // world_size
    row_rank0 = RowParallelLinear(
        in_features, out_features, bias=False, rank=0, world_size=world_size, communicator=comm0
    )
    row_rank1 = RowParallelLinear(
        in_features, out_features, bias=False, rank=1, world_size=world_size, communicator=comm1
    )

    with torch.no_grad():
        # Transpose full_linear.weight: [out_features, in_features] -> [in_features, out_features]
        w_t = full_linear.weight.t()
        row_rank0.weight.copy_(w_t[0:chunk_size, :])
        row_rank1.weight.copy_(w_t[chunk_size:in_features, :])

        x0 = x[:, :, 0:chunk_size]
        x1 = x[:, :, chunk_size:in_features]

        # In parallel forward step
        # Simulate rank 0 and 1 exchange
        import threading
        res = [None, None]

        def run_rank0():
            res[0] = row_rank0(x0, step_tag="test_row_parallel")

        def run_rank1():
            res[1] = row_rank1(x1, step_tag="test_row_parallel")

        t0 = threading.Thread(target=run_rank0)
        t1 = threading.Thread(target=run_rank1)
        t0.start()
        t1.start()
        t0.join()
        t1.join()

    assert torch.allclose(expected_full_out, res[0], atol=1e-5)
    assert torch.allclose(expected_full_out, res[1], atol=1e-5)


def test_rmsnorm():
    """Verifies RMSNorm numerical correctness."""
    dim = 64
    norm = RMSNorm(dim, eps=1e-5)
    x = torch.randn(2, 5, dim)
    y = norm(x)

    assert y.shape == x.shape
    # Check that variance per token is close to 1
    var = y.pow(2).mean(dim=-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)
