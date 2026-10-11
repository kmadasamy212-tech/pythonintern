"""
Tensor Parallelism Layers for Sharded Transformer Execution.
Implements ColumnParallelLinear, RowParallelLinear, and RMSNorm
following Megatron-LM principles adapted for decentralized gRPC collectives.
"""

import math
from typing import Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object

from engine.tensor_parallel.comm import DistributedCommunicator


class ColumnParallelLinear(nn.Module if HAS_TORCH else object):
    """
    Linear layer partitioned across output features (columns of weight matrix).
    Weight: [in_features, out_features // world_size]
    Forward pass computes local chunk of output features without communication:
        Y_local = X @ W_local + b_local
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        rank: int = 0,
        world_size: int = 1,
        gather_output: bool = False,
        dtype=None,
    ):
        if HAS_TORCH:
            super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.gather_output = gather_output

        assert out_features % world_size == 0, (
            f"out_features ({out_features}) must be divisible by world_size ({world_size})"
        )
        self.out_features_per_partition = out_features // world_size

        if HAS_TORCH:
            self.weight = nn.Parameter(
                torch.empty(
                    self.in_features,
                    self.out_features_per_partition,
                    dtype=dtype or torch.float32,
                )
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(self.out_features_per_partition, dtype=dtype or torch.float32)
                )
            else:
                self.register_parameter("bias", None)
            self.reset_parameters()

    def reset_parameters(self):
        if HAS_TORCH:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        x: [batch, seq_len, in_features]
        returns: [batch, seq_len, out_features_per_partition]
        """
        output = torch.matmul(x, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output


class RowParallelLinear(nn.Module if HAS_TORCH else object):
    """
    Linear layer partitioned across input features (rows of weight matrix).
    Weight: [in_features // world_size, out_features]
    Forward pass computes partial output:
        Y_partial = X_local @ W_local
    followed by an All-Reduce SUM across all worker ranks:
        Y = AllReduce(Y_partial) + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        rank: int = 0,
        world_size: int = 1,
        communicator: Optional[DistributedCommunicator] = None,
        dtype=None,
    ):
        if HAS_TORCH:
            super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.communicator = communicator

        assert in_features % world_size == 0, (
            f"in_features ({in_features}) must be divisible by world_size ({world_size})"
        )
        self.in_features_per_partition = in_features // world_size

        if HAS_TORCH:
            self.weight = nn.Parameter(
                torch.empty(
                    self.in_features_per_partition,
                    self.out_features,
                    dtype=dtype or torch.float32,
                )
            )
            if bias:
                self.bias = nn.Parameter(torch.empty(self.out_features, dtype=dtype or torch.float32))
            else:
                self.register_parameter("bias", None)
            self.reset_parameters()

    def reset_parameters(self):
        if HAS_TORCH:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

    def set_communicator(self, communicator: DistributedCommunicator):
        self.communicator = communicator

    def forward(self, x: "torch.Tensor", step_tag: str = "") -> "torch.Tensor":
        """
        x: [batch, seq_len, in_features_per_partition]
        returns: [batch, seq_len, out_features]
        """
        partial_output = torch.matmul(x, self.weight)

        # Collective communication: All-Reduce (SUM) partial sums across all workers
        if self.world_size > 1 and self.communicator is not None:
            output = self.communicator.all_reduce_sum(partial_output, step_tag=step_tag)
        else:
            output = partial_output

        if self.bias is not None:
            output = output + self.bias
        return output


class RMSNorm(nn.Module if HAS_TORCH else object):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    Computed locally on each rank since hidden_dim is fully gathered.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        if HAS_TORCH:
            super().__init__()
        self.dim = dim
        self.eps = eps
        if HAS_TORCH:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if not HAS_TORCH:
            return x
        variance = x.pow(2).mean(-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        return norm_x * self.weight
