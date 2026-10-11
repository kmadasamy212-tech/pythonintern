"""
Sharded SwiGLU Feed-Forward Network (MLP).
Uses ColumnParallelLinear for Gate & Up projections and RowParallelLinear for Down projection
with gRPC All-Reduce to minimize decentralized network overhead.
"""

from typing import Optional
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


class DistributedMLP(nn.Module if HAS_TORCH else object):
    """
    SwiGLU Multi-Layer Perceptron partitioned across tensor-parallel workers:
        Gate = X @ W_gate_local
        Up   = X @ W_up_local
        H    = (SiLU(Gate) * Up)
        Out  = AllReduce(H @ W_down_local)
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
        self.communicator = communicator

        assert config.intermediate_dim % world_size == 0, (
            f"intermediate_dim {config.intermediate_dim} must be divisible by world_size {world_size}"
        )

        # Gate and Up projections (Column-Parallel)
        self.gate_proj = ColumnParallelLinear(
            config.hidden_dim, config.intermediate_dim, bias=False, rank=rank, world_size=world_size
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_dim, config.intermediate_dim, bias=False, rank=rank, world_size=world_size
        )
        # Down projection (Row-Parallel with All-Reduce)
        self.down_proj = RowParallelLinear(
            config.intermediate_dim,
            config.hidden_dim,
            bias=False,
            rank=rank,
            world_size=world_size,
            communicator=communicator,
        )

    def set_communicator(self, communicator: DistributedCommunicator):
        self.communicator = communicator
        self.down_proj.set_communicator(communicator)

    def forward(self, x: "torch.Tensor", step_id: int = 0) -> "torch.Tensor":
        """
        x: [batch, seq_len, hidden_dim]
        returns: [batch, seq_len, hidden_dim]
        """
        if not HAS_TORCH:
            return x

        gate = self.gate_proj(x)
        up = self.up_proj(x)
        intermediate = F.silu(gate) * up

        step_tag = f"step_{step_id}_l{self.layer_idx}_mlp"
        output = self.down_proj(intermediate, step_tag=step_tag)
        return output
