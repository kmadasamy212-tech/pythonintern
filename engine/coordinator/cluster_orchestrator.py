"""
Cluster Orchestrator for Distributed Tensor Parallelism.
Manages gRPC communication across all worker shards, coordinating step dispatches and collective syncs.
"""

from concurrent import futures
import logging
from typing import Dict, List, Optional, Tuple
import grpc

from engine.config import ClusterConfig, ModelConfig, ShardConfig, KVCacheConfig
from engine.scheduler.continuous_batcher import BatchSchedule
import proto_gen.distributed_inference_pb2 as pb2
import proto_gen.distributed_inference_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


class ClusterOrchestrator:
    """
    Coordinates execution across the distributed cluster of commodity nodes.
    """

    def __init__(self, cluster_config: ClusterConfig):
        self.config = cluster_config
        self.worker_endpoints = cluster_config.worker_endpoints
        self.world_size = len(self.worker_endpoints)
        self.channels: List[grpc.Channel] = []
        self.stubs: List[pb2_grpc.WorkerServiceStub] = []
        self.executor = futures.ThreadPoolExecutor(max_workers=max(4, self.world_size * 2))

    def connect(self):
        """Establishes gRPC channels to all worker nodes."""
        self.channels.clear()
        self.stubs.clear()
        for ep in self.worker_endpoints:
            channel = grpc.insecure_channel(
                ep,
                options=[
                    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                    ("grpc.max_send_message_length", 100 * 1024 * 1024),
                ],
            )
            stub = pb2_grpc.WorkerServiceStub(channel)
            self.channels.append(channel)
            self.stubs.append(stub)

    def initialize_shards(self) -> bool:
        """Sends InitializeShard requests to all worker nodes in parallel."""
        m_proto = pb2.ModelConfigProto(
            vocab_size=self.config.model.vocab_size,
            hidden_dim=self.config.model.hidden_dim,
            num_layers=self.config.model.num_layers,
            num_heads=self.config.model.num_heads,
            num_kv_heads=self.config.model.num_kv_heads,
            intermediate_dim=self.config.model.intermediate_dim,
            max_seq_len=self.config.model.max_seq_len,
            rms_norm_eps=self.config.model.rms_norm_eps,
        )

        def init_worker(rank: int):
            shard_proto = pb2.ShardConfigProto(
                rank=rank,
                world_size=self.world_size,
                peer_addresses=self.worker_endpoints,
            )
            req = pb2.InitializeShardRequest(
                model_config=m_proto,
                shard_config=shard_proto,
                num_kv_blocks=self.config.kv_cache.num_blocks,
                block_size=self.config.kv_cache.block_size,
            )
            try:
                resp = self.stubs[rank].InitializeShard(req, timeout=30.0)
                logger.info(f"Worker {rank} initialized: {resp.message}")
                return resp.success
            except Exception as e:
                logger.error(f"Failed to initialize worker {rank}: {e}")
                return False

        results = list(self.executor.map(init_worker, range(self.world_size)))
        return all(results)

    def execute_step(self, schedule: BatchSchedule) -> Dict[str, int]:
        """
        Dispatches StepRequest to all worker shards concurrently.
        Gathers next sampled token IDs for all active sequences in the batch.
        """
        active_slots = []

        # Prefills
        for seq in schedule.prefill_sequences:
            slot = pb2.SequenceSlotProto(
                seq_id=seq.seq_id,
                is_prefill=True,
                prompt_token_ids=seq.prompt_token_ids,
                current_len=len(seq.prompt_token_ids),
                block_table=schedule.block_tables.get(seq.seq_id, []),
            )
            active_slots.append(slot)

        # Decodes
        for seq in schedule.decode_sequences:
            slot = pb2.SequenceSlotProto(
                seq_id=seq.seq_id,
                is_prefill=False,
                current_token_id=seq.current_token_id,
                current_len=seq.total_length,
                block_table=schedule.block_tables.get(seq.seq_id, []),
            )
            active_slots.append(slot)

        step_req = pb2.StepRequest(
            step_id=schedule.step_id,
            active_sequences=active_slots,
        )

        def run_worker_step(rank: int) -> pb2.StepResponse:
            try:
                return self.stubs[rank].ExecuteStep(step_req, timeout=self.config.scheduler.step_timeout_sec)
            except Exception as e:
                logger.error(f"Error executing step on worker {rank}: {e}")
                return pb2.StepResponse(step_id=schedule.step_id, rank=rank, success=False, error_message=str(e))

        # Run simultaneously across all workers
        responses = list(self.executor.map(run_worker_step, range(self.world_size)))

        # Collect sampled tokens from rank 0
        sampled_tokens: Dict[str, int] = {}
        rank0_resp = next((r for r in responses if r.rank == 0 and r.success), None)
        if rank0_resp:
            for s in rank0_resp.sampled_tokens:
                sampled_tokens[s.seq_id] = s.token_id

        return sampled_tokens

    def evict_sequences(self, seq_ids: List[str]):
        """Notifies all workers to evict finished sequences from their KV cache pools."""
        if not seq_ids:
            return
        req = pb2.EvictSequencesRequest(seq_ids=seq_ids)

        def evict_worker(rank: int):
            try:
                self.stubs[rank].EvictSequences(req, timeout=5.0)
            except Exception as e:
                logger.warning(f"Error evicting sequences on worker {rank}: {e}")

        list(self.executor.map(evict_worker, range(self.world_size)))

    def get_cluster_status(self) -> pb2.ClusterStatusResponse:
        """Polls status from all workers."""
        worker_statuses = []
        healthy_count = 0

        req = pb2.WorkerStatusRequest()
        for stub in self.stubs:
            try:
                stat = stub.GetStatus(req, timeout=3.0)
                worker_statuses.append(stat)
                if stat.is_ready:
                    healthy_count += 1
            except Exception:
                pass

        return pb2.ClusterStatusResponse(
            num_workers=self.world_size,
            healthy_workers=healthy_count,
            worker_statuses=worker_statuses,
        )

    def close(self):
        for ch in self.channels:
            try:
                ch.close()
            except Exception:
                pass
        self.executor.shutdown(wait=False)
