"""
Configuration module for the distributed inference engine.
Defines model architecture, tensor parallel cluster topology,
KV cache memory paging, and continuous batching scheduler parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class ModelConfig:
    """Hyperparameters for sharded Transformer LLM (LLaMA/Mistral style)."""
    vocab_size: int = 32000
    hidden_dim: int = 1024
    num_layers: int = 8
    num_heads: int = 16
    num_kv_heads: int = 4            # Grouped Query Attention (GQA)
    intermediate_dim: int = 2816     # SwiGLU hidden dim (approx 8/3 * hidden_dim)
    max_seq_len: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dtype: str = "float32"            # "float32", "float16", "bfloat16"

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_heads // self.num_kv_heads

    def validate(self):
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"


@dataclass
class ShardConfig:
    """Distributed Tensor Parallel node configuration."""
    rank: int = 0
    world_size: int = 1
    host: str = "127.0.0.1"
    port: int = 50051
    peer_addresses: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class KVCacheConfig:
    """Paged KV cache configuration for commodity memory efficiency."""
    block_size: int = 16            # Number of tokens stored per physical block
    num_blocks: int = 256           # Total pool of cache blocks on this node
    device: str = "cpu"             # "cpu" or "cuda"

    @property
    def total_token_capacity(self) -> int:
        return self.block_size * self.num_blocks


@dataclass
class SchedulerConfig:
    """Continuous Batching Iteration-Level Scheduler configuration."""
    max_batch_size: int = 16
    max_tokens_per_step: int = 512
    step_timeout_sec: float = 10.0
    preemption_mode: str = "recompute"  # "recompute" or "swap"


@dataclass
class ClusterConfig:
    """Full cluster specification."""
    coordinator_host: str = "127.0.0.1"
    coordinator_port: int = 50050
    worker_endpoints: List[str] = field(default_factory=lambda: ["127.0.0.1:50051", "127.0.0.1:50052"])
    model: ModelConfig = field(default_factory=ModelConfig)
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @property
    def world_size(self) -> int:
        return len(self.worker_endpoints)
