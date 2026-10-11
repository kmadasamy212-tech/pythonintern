"""
Worker Node gRPC Server.
Hosts a single tensor-parallel LLM shard, handles peer-to-peer collective exchanges,
and executes forward steps commanded by the continuous batching coordinator.
"""

from concurrent import futures
import logging
import time
from typing import Optional
import grpc

from engine.config import ModelConfig, ShardConfig, KVCacheConfig
from engine.tensor_parallel.comm import DistributedCommunicator
from engine.tensor_parallel.buffer_codec import deserialize_tensor
from engine.worker.execution_engine import WorkerExecutionEngine
import proto_gen.distributed_inference_pb2 as pb2
import proto_gen.distributed_inference_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


class WorkerServiceServicer(pb2_grpc.WorkerServiceServicer):
    """Implementation of WorkerService for tensor parallel shards."""

    def __init__(self, rank: int, world_size: int, host: str = "127.0.0.1", port: int = 50051):
        self.rank = rank
        self.world_size = world_size
        self.host = host
        self.port = port
        self.communicator = DistributedCommunicator(rank=rank, world_size=world_size)
        self.engine: Optional[WorkerExecutionEngine] = None
        self.is_ready = False

    def InitializeShard(self, request: pb2.InitializeShardRequest, context) -> pb2.InitializeShardResponse:
        logger.info(f"[Rank {self.rank}] Initializing shard with world_size {self.world_size}...")
        try:
            m_proto = request.model_config
            model_config = ModelConfig(
                vocab_size=m_proto.vocab_size or 32000,
                hidden_dim=m_proto.hidden_dim or 1024,
                num_layers=m_proto.num_layers or 8,
                num_heads=m_proto.num_heads or 16,
                num_kv_heads=m_proto.num_kv_heads or 4,
                intermediate_dim=m_proto.intermediate_dim or 2816,
                max_seq_len=m_proto.max_seq_len or 2048,
                rms_norm_eps=m_proto.rms_norm_eps or 1e-5,
            )

            shard_config = ShardConfig(
                rank=self.rank,
                world_size=self.world_size,
                host=self.host,
                port=self.port,
                peer_addresses=list(request.shard_config.peer_addresses),
            )

            kv_config = KVCacheConfig(
                block_size=request.block_size or 16,
                num_blocks=request.num_kv_blocks or 256,
            )

            # Establish peer gRPC channels for All-Reduce collectives
            for idx, peer_addr in enumerate(shard_config.peer_addresses):
                if peer_addr and idx != self.rank:
                    channel = grpc.insecure_channel(peer_addr)
                    stub = pb2_grpc.WorkerServiceStub(channel)
                    self.communicator.register_peer_stub(idx, stub)
                    self.communicator._channels[idx] = channel

            self.engine = WorkerExecutionEngine(
                model_config=model_config,
                shard_config=shard_config,
                kv_config=kv_config,
                communicator=self.communicator,
            )
            self.is_ready = True

            return pb2.InitializeShardResponse(
                success=True,
                message=f"Shard {self.rank}/{self.world_size} initialized successfully.",
                allocated_memory_bytes=0,
            )
        except Exception as e:
            logger.error(f"[Rank {self.rank}] Initialization failed: {e}", exc_info=True)
            return pb2.InitializeShardResponse(success=False, message=str(e))

    def ExecuteStep(self, request: pb2.StepRequest, context) -> pb2.StepResponse:
        if not self.is_ready or self.engine is None:
            return pb2.StepResponse(
                step_id=request.step_id,
                rank=self.rank,
                success=False,
                error_message="Worker shard not initialized",
            )
        try:
            return self.engine.execute_step(request)
        except Exception as e:
            logger.error(f"[Rank {self.rank}] Step execution error: {e}", exc_info=True)
            return pb2.StepResponse(
                step_id=request.step_id,
                rank=self.rank,
                success=False,
                error_message=str(e),
            )

    def ExchangeAllReduce(self, request: pb2.AllReducePayload, context) -> pb2.AllReduceResponse:
        """
        Receives partial tensor slices from peer worker nodes for collective All-Reduce.
        """
        try:
            tensor = deserialize_tensor(
                raw_bytes=request.raw_tensor_data,
                shape=tuple(request.shape),
                dtype_str=request.dtype,
            )
            self.communicator.handle_incoming_allreduce(
                step_tag=request.step_tag,
                sender_rank=request.sender_rank,
                tensor=tensor,
            )
            return pb2.AllReduceResponse(success=True)
        except Exception as e:
            logger.error(f"[Rank {self.rank}] ExchangeAllReduce error: {e}")
            return pb2.AllReduceResponse(success=False, error_message=str(e))

    def EvictSequences(self, request: pb2.EvictSequencesRequest, context) -> pb2.EvictSequencesResponse:
        if self.engine is None:
            return pb2.EvictSequencesResponse(success=False, freed_blocks=0)
        freed = self.engine.evict_sequences(list(request.seq_ids))
        return pb2.EvictSequencesResponse(success=True, freed_blocks=freed)

    def GetStatus(self, request: pb2.WorkerStatusRequest, context) -> pb2.WorkerStatusResponse:
        free_blocks = self.engine.kv_manager.free_block_count if self.engine else 0
        total_blocks = self.engine.kv_manager.total_blocks if self.engine else 0
        avg_lat = (
            (self.engine.total_latency_ms / max(1, self.engine.total_steps_executed))
            if self.engine
            else 0.0
        )

        return pb2.WorkerStatusResponse(
            rank=self.rank,
            world_size=self.world_size,
            is_ready=self.is_ready,
            memory_used_bytes=0,
            total_memory_bytes=0,
            total_kv_blocks=total_blocks,
            free_kv_blocks=free_blocks,
            avg_step_latency_ms=avg_lat,
            device="cpu",
        )


def serve_worker(rank: int, world_size: int, host: str = "127.0.0.1", port: int = 50051) -> grpc.Server:
    """Starts the gRPC worker server."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=16),
        options=[
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ],
    )
    servicer = WorkerServiceServicer(rank=rank, world_size=world_size, host=host, port=port)
    pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info(f"Worker Rank {rank} running on {host}:{port}")
    return server
