"""
End-to-End Integration Test for Distributed LLM Inference Engine.
Spins up 2 worker gRPC servers and 1 coordinator server, executes streaming inference,
and shuts down cleanly.
"""

import asyncio
import time
import pytest

from engine.config import ClusterConfig, ModelConfig, KVCacheConfig, SchedulerConfig
from engine.worker.worker_node import serve_worker
from engine.coordinator.cluster_orchestrator import ClusterOrchestrator
from engine.coordinator.coordinator import serve_coordinator_async
from engine.client.grpc_client import DistributedLLMClient


async def _run_e2e_test():
    world_size = 2
    worker_ports = [50061, 50062]
    coord_port = 50060

    # 1. Start Worker Servers
    worker_servers = []
    for rank in range(world_size):
        srv = serve_worker(rank=rank, world_size=world_size, host="127.0.0.1", port=worker_ports[rank])
        worker_servers.append(srv)

    time.sleep(0.5)

    try:
        # 2. Configure Cluster
        model_cfg = ModelConfig(
            vocab_size=500,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate_dim=256,
            max_seq_len=256,
        )
        cluster_cfg = ClusterConfig(
            coordinator_host="127.0.0.1",
            coordinator_port=coord_port,
            worker_endpoints=[f"127.0.0.1:{p}" for p in worker_ports],
            model=model_cfg,
            kv_cache=KVCacheConfig(block_size=8, num_blocks=64),
            scheduler=SchedulerConfig(max_batch_size=4),
        )

        orchestrator = ClusterOrchestrator(cluster_cfg)
        orchestrator.connect()

        init_success = orchestrator.initialize_shards()
        assert init_success, "Failed to initialize worker shards"

        # 3. Start Coordinator
        coord_server = await serve_coordinator_async(
            cluster_config=cluster_cfg,
            orchestrator=orchestrator,
            host="127.0.0.1",
            port=coord_port,
        )

        time.sleep(0.5)

        # 4. Client Request Test
        client = DistributedLLMClient(target=f"127.0.0.1:{coord_port}")

        loop = asyncio.get_running_loop()

        # Check status (run in threadpool so as not to block coordinator event loop)
        status = await loop.run_in_executor(None, client.get_cluster_status)
        assert status.num_workers == 2
        assert status.healthy_workers == 2

        # Streaming generation test (run in threadpool)
        def run_streaming():
            return list(
                client.generate_stream(
                    prompt="Distributed inference over commodity nodes",
                    max_new_tokens=4,
                    temperature=0.0,
                )
            )

        responses = await loop.run_in_executor(None, run_streaming)

        assert len(responses) > 0
        assert responses[-1].is_finished
        assert responses[-1].total_tokens_generated >= 1

        client.close()
        await coord_server.stop(0)
        orchestrator.close()
    finally:
        for srv in worker_servers:
            srv.stop(0)


def test_end_to_end_distributed_inference():
    asyncio.run(_run_e2e_test())
