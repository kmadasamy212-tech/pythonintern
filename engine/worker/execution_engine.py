"""
Worker Execution Engine.
Manages local model shard forward passes, KV cache reads/writes, and local layer reductions.
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from engine.config import ModelConfig, ShardConfig, KVCacheConfig
from engine.model.llama import DistributedLLaMAModel
from engine.model.kv_cache import KVCacheManager
from engine.tensor_parallel.comm import DistributedCommunicator
import proto_gen.distributed_inference_pb2 as pb2


class WorkerExecutionEngine:
    """
    Executes sharded model computation on a single commodity worker node.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        shard_config: ShardConfig,
        kv_config: KVCacheConfig,
        communicator: DistributedCommunicator,
        device: str = "cpu",
    ):
        self.model_config = model_config
        self.shard_config = shard_config
        self.kv_config = kv_config
        self.communicator = communicator
        self.device = device

        # Initialize KV Cache Manager for local shard
        self.kv_manager = KVCacheManager(
            config=kv_config,
            model_config=model_config,
            world_size=shard_config.world_size,
        )

        # Initialize Sharded LLaMA Model
        self.model = DistributedLLaMAModel(
            config=model_config,
            rank=shard_config.rank,
            world_size=shard_config.world_size,
            communicator=communicator,
            device=device,
        )

        # Performance stats
        self.total_steps_executed = 0
        self.total_latency_ms = 0.0

    def execute_step(self, step_request: pb2.StepRequest) -> pb2.StepResponse:
        """
        Executes a continuous batching iteration step.
        Dispatches prefill sequences and decode sequences through sharded transformer.
        """
        start_time = time.perf_counter()
        step_id = step_request.step_id
        sampled_tokens = []

        prefill_slots = [s for s in step_request.active_sequences if s.is_prefill]
        decode_slots = [s for s in step_request.active_sequences if not s.is_prefill]

        if not HAS_TORCH:
            return pb2.StepResponse(
                step_id=step_id,
                rank=self.shard_config.rank,
                success=False,
                error_message="PyTorch not available on worker",
            )

        with torch.no_grad():
            # 1. Execute Prefill Sequences
            for slot in prefill_slots:
                prompt_tokens = torch.tensor([list(slot.prompt_token_ids)], dtype=torch.long, device=self.device)
                seq_len = prompt_tokens.shape[1]
                positions = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
                block_table = [list(slot.block_table)]

                logits = self.model(
                    token_ids=prompt_tokens,
                    positions=positions,
                    kv_pool=self.kv_manager.pool,
                    block_tables=block_table,
                    is_prefill=True,
                    step_id=step_id,
                    return_logits_last_only=True,
                )

                # Rank 0 (or all ranks) produces next sampled token (Greedy by default on worker,
                # coordinator can resample if full logits needed)
                token_id = int(torch.argmax(logits[0]).item())
                sampled_tokens.append(
                    pb2.SampledTokenProto(
                        seq_id=slot.seq_id,
                        token_id=token_id,
                        top_logits=[float(x) for x in logits[0][:10]],
                    )
                )

            # 2. Execute Decode Sequences (Batched)
            if decode_slots:
                batch_size = len(decode_slots)
                token_ids_list = [[slot.current_token_id] for slot in decode_slots]
                positions_list = [[slot.current_len - 1] for slot in decode_slots]
                block_tables = [list(slot.block_table) for slot in decode_slots]

                token_ids_tensor = torch.tensor(token_ids_list, dtype=torch.long, device=self.device)
                positions_tensor = torch.tensor(positions_list, dtype=torch.long, device=self.device)

                logits = self.model(
                    token_ids=token_ids_tensor,
                    positions=positions_tensor,
                    kv_pool=self.kv_manager.pool,
                    block_tables=block_tables,
                    is_prefill=False,
                    step_id=step_id,
                    return_logits_last_only=True,
                )

                for idx, slot in enumerate(decode_slots):
                    token_id = int(torch.argmax(logits[idx]).item())
                    sampled_tokens.append(
                        pb2.SampledTokenProto(
                            seq_id=slot.seq_id,
                            token_id=token_id,
                            top_logits=[float(x) for x in logits[idx][:10]],
                        )
                    )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.total_steps_executed += 1
        self.total_latency_ms += elapsed_ms

        return pb2.StepResponse(
            step_id=step_id,
            rank=self.shard_config.rank,
            success=True,
            sampled_tokens=sampled_tokens,
            step_latency_ms=elapsed_ms,
            active_blocks=self.kv_manager.allocated_block_count,
            free_blocks=self.kv_manager.free_block_count,
        )

    def evict_sequences(self, seq_ids: List[str]) -> int:
        freed = 0
        for sid in seq_ids:
            freed += self.kv_manager.free_sequence(sid)
        return freed
