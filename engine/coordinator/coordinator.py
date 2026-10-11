"""
Inference Coordinator Server.
Hosts the client-facing gRPC API, runs the continuous batching event loop,
and streams generated tokens asynchronously back to clients.
"""

import asyncio
from concurrent import futures
import logging
import time
from typing import AsyncIterator, Dict, List, Optional
import uuid
import grpc

from engine.config import ClusterConfig
from engine.scheduler.sequence import Sequence, SequenceStatus
from engine.scheduler.continuous_batcher import ContinuousBatcher
from engine.coordinator.cluster_orchestrator import ClusterOrchestrator
import proto_gen.distributed_inference_pb2 as pb2
import proto_gen.distributed_inference_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


class SimpleTokenizer:
    """Lightweight byte/character tokenizer for realistic standalone testing."""

    def encode(self, text: str) -> List[int]:
        # Simple ASCII / UTF-8 token mapping + offset
        return [b + 10 for b in text.encode("utf-8")]

    def decode(self, token_ids: List[int]) -> str:
        bytes_list = bytearray()
        for t in token_ids:
            if t >= 10 and (t - 10) < 256:
                bytes_list.append(t - 10)
        return bytes_list.decode("utf-8", errors="replace")


class InferenceCoordinatorServicer(pb2_grpc.InferenceCoordinatorServiceServicer):
    """
    Client-facing coordinator service with continuous batching and token streaming.
    """

    def __init__(self, cluster_config: ClusterConfig, orchestrator: ClusterOrchestrator):
        self.config = cluster_config
        self.orchestrator = orchestrator
        self.batcher = ContinuousBatcher(
            scheduler_config=cluster_config.scheduler,
            kv_config=cluster_config.kv_cache,
        )
        self.tokenizer = SimpleTokenizer()
        self.is_running = True
        self._loop_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.total_tokens_generated = 0
        self.start_time = time.perf_counter()

    def start_scheduler_thread(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._loop_task = loop.create_task(self._run_continuous_loop())

    async def _run_continuous_loop(self):
        """Continuous iteration-level scheduling loop."""
        logger.info("Continuous batching scheduler loop started.")
        while self.is_running:
            if not self.batcher.waiting_queue and not self.batcher.running_sequences:
                await asyncio.sleep(0.005)
                continue

            schedule = self.batcher.schedule_step()
            if schedule.is_empty:
                await asyncio.sleep(0.005)
                continue

            # Execute forward step across worker shards in threadpool
            sampled_tokens = await self._loop.run_in_executor(
                None, self.orchestrator.execute_step, schedule
            )

            # Advance batch sequences
            results = self.batcher.process_step_outputs(schedule, sampled_tokens)

            # Route tokens to streaming client queues
            finished_ids = []
            for seq, token_id, is_finished in results:
                self.total_tokens_generated += 1
                if seq.stream_queue is not None:
                    delta_text = self.tokenizer.decode([token_id])
                    await seq.stream_queue.put((token_id, delta_text, is_finished, seq.status.value))

                if is_finished:
                    finished_ids.append(seq.seq_id)

            # Evict finished sequences from worker KV caches
            if finished_ids:
                await self._loop.run_in_executor(
                    None, self.orchestrator.evict_sequences, finished_ids
                )

    async def GenerateStream(
        self, request: pb2.GenerateRequest, context
    ) -> AsyncIterator[pb2.GenerateResponse]:
        """
        Streaming generation API for clients.
        Yields tokens one-by-one as continuous batcher produces them.
        """
        req_id = request.request_id or str(uuid.uuid4())
        prompt_tokens = (
            list(request.prompt_token_ids)
            if request.prompt_token_ids
            else self.tokenizer.encode(request.prompt)
        )
        if not prompt_tokens:
            prompt_tokens = [1]  # BOS token fallback

        sampling = request.sampling_params
        seq = Sequence(
            seq_id=req_id,
            prompt=request.prompt,
            prompt_token_ids=prompt_tokens,
            max_new_tokens=sampling.max_new_tokens or 32,
            temperature=sampling.temperature if sampling.temperature > 0 else 0.0,
            top_p=sampling.top_p or 0.9,
            top_k=sampling.top_k or 50,
            stop_token_ids=list(sampling.stop_token_ids) if sampling.stop_token_ids else [0, 2],
        )

        queue = asyncio.Queue()
        seq.stream_queue = queue

        # Enqueue sequence into continuous batcher
        self.batcher.add_request(seq)

        start_time = time.perf_counter()
        first_token = True
        ttft_ms = 0.0
        generated_count = 0

        while True:
            try:
                token_id, delta_text, is_finished, finish_reason = await asyncio.wait_for(
                    queue.get(), timeout=30.0
                )
            except asyncio.TimeoutError:
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "Generation timed out")
                break

            generated_count += 1
            now = time.perf_counter()
            if first_token:
                ttft_ms = (now - start_time) * 1000.0
                first_token = False

            total_lat = (now - start_time) * 1000.0

            yield pb2.GenerateResponse(
                request_id=req_id,
                token_id=token_id,
                text_delta=delta_text,
                is_finished=is_finished,
                finish_reason=finish_reason if is_finished else "",
                ttft_ms=ttft_ms,
                total_latency_ms=total_lat,
                total_tokens_generated=generated_count,
            )

            if is_finished:
                break

    def GetClusterStatus(
        self, request: pb2.ClusterStatusRequest, context
    ) -> pb2.ClusterStatusResponse:
        status = self.orchestrator.get_cluster_status()
        status.queue_depth = len(self.batcher.waiting_queue)
        status.running_requests = len(self.batcher.running_sequences)
        status.total_tokens_generated = self.total_tokens_generated
        elapsed = max(0.1, time.perf_counter() - self.start_time)
        status.current_throughput_tokens_sec = self.total_tokens_generated / elapsed
        return status


async def serve_coordinator_async(
    cluster_config: ClusterConfig,
    orchestrator: ClusterOrchestrator,
    host: str = "127.0.0.1",
    port: int = 50050,
) -> grpc.aio.Server:
    """Starts the asynchronous gRPC coordinator server."""
    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ]
    )
    servicer = InferenceCoordinatorServicer(cluster_config, orchestrator)
    pb2_grpc.add_InferenceCoordinatorServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")

    loop = asyncio.get_running_loop()
    servicer.start_scheduler_thread(loop)

    await server.start()
    logger.info(f"Inference Coordinator running on {host}:{port}")
    return server
