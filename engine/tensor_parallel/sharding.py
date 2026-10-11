"""
Weight sharding and parameter partitioning module.
Splits full model state dicts into rank-specific shards for commodity workers.
"""

from typing import Dict, Any, Tuple
import math

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from engine.config import ModelConfig


class WeightSharder:
    """
    Partitions full model weights into tensor-parallel shards across world_size workers.
    """

    def __init__(self, model_config: ModelConfig, world_size: int):
        self.config = model_config
        self.world_size = world_size
        self.heads_per_rank = model_config.num_heads // world_size
        self.kv_heads_per_rank = model_config.num_kv_heads // world_size
        self.inter_dim_per_rank = model_config.intermediate_dim // world_size

    def shard_column_parallel_weight(self, weight: "torch.Tensor", rank: int) -> "torch.Tensor":
        """
        Weight shape: [in_features, out_features]
        Col-parallel slices along dimension 1 (columns).
        """
        out_features = weight.shape[1]
        chunk_size = out_features // self.world_size
        start = rank * chunk_size
        end = (rank + 1) * chunk_size
        return weight[:, start:end].clone()

    def shard_row_parallel_weight(self, weight: "torch.Tensor", rank: int) -> "torch.Tensor":
        """
        Weight shape: [in_features, out_features]
        Row-parallel slices along dimension 0 (rows).
        """
        in_features = weight.shape[0]
        chunk_size = in_features // self.world_size
        start = rank * chunk_size
        end = (rank + 1) * chunk_size
        return weight[start:end, :].clone()

    def shard_attention_weights(
        self,
        q_weight: "torch.Tensor",
        k_weight: "torch.Tensor",
        v_weight: "torch.Tensor",
        o_weight: "torch.Tensor",
        rank: int,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Shards Q, K, V as ColumnParallel and O as RowParallel.
        """
        # Q: [hidden_dim, num_heads * head_dim] -> slice out local heads
        q_shard = self.shard_column_parallel_weight(q_weight, rank)
        # K, V: [hidden_dim, num_kv_heads * head_dim] -> slice out local kv heads
        k_shard = self.shard_column_parallel_weight(k_weight, rank)
        v_shard = self.shard_column_parallel_weight(v_weight, rank)
        # O: [num_heads * head_dim, hidden_dim] -> slice out local input heads
        o_shard = self.shard_row_parallel_weight(o_weight, rank)
        return q_shard, k_shard, v_shard, o_shard

    def shard_mlp_weights(
        self,
        gate_weight: "torch.Tensor",
        up_weight: "torch.Tensor",
        down_weight: "torch.Tensor",
        rank: int,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Shards SwiGLU MLP: gate & up as ColumnParallel, down as RowParallel.
        """
        gate_shard = self.shard_column_parallel_weight(gate_weight, rank)
        up_shard = self.shard_column_parallel_weight(up_weight, rank)
        down_shard = self.shard_row_parallel_weight(down_weight, rank)
        return gate_shard, up_shard, down_shard

    def shard_full_model_state_dict(
        self, full_state_dict: Dict[str, Any], rank: int
    ) -> Dict[str, Any]:
        """
        Takes the monolithic state_dict of an entire LLM and extracts the exact shard
        parameters for the specified rank.
        """
        sharded_dict = {}
        for key, tensor in full_state_dict.items():
            if "q_proj.weight" in key or "k_proj.weight" in key or "v_proj.weight" in key:
                sharded_dict[key] = self.shard_column_parallel_weight(tensor, rank)
            elif "out_proj.weight" in key:
                sharded_dict[key] = self.shard_row_parallel_weight(tensor, rank)
            elif "gate_proj.weight" in key or "up_proj.weight" in key:
                sharded_dict[key] = self.shard_column_parallel_weight(tensor, rank)
            elif "down_proj.weight" in key:
                sharded_dict[key] = self.shard_row_parallel_weight(tensor, rank)
            else:
                # Norm weights, embeddings, LM head replicated or handled directly
                sharded_dict[key] = tensor.clone()
        return sharded_dict
