"""
Cluster Launcher Script.
Orchestrates the startup of N distributed tensor-parallel commodity worker nodes
and the central continuous batching inference coordinator.
"""

import argparse
import asyncio
import logging
import multiprocessing
import os
import sys
import time
from typing import List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.config import ClusterConfig, ModelConfig, KVCacheConfig, SchedulerConfig
from engine.worker.worker_node import serve_worker
from engine.coordinator.cluster_orchestrator import ClusterOrchestrator
from engine.coordinator.coordinator import serve_coordinator_async

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ClusterLauncher")


def run_worker_process(rank: int, world_size: int, host: str, port: int):
    """Entrypoint for a single worker node process."""
    logger.info(f"Starting Worker Node Rank {rank}/{world_size} on {host}:{port}...")
    server = serve_worker(rank=rank, world_size=world_size, host=host, port=port)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


async def run_coordinator_async(cluster_config: ClusterConfig):
    """Entrypoint for the Coordinator server."""
    logger.info(f"Connecting Coordinator to {cluster_config.world_size} worker nodes...")
    orchestrator = ClusterOrchestrator(cluster_config)
    orchestrator.connect()

    # Wait for workers to be reachable
    logger.info("Initializing model shards across commodity workers...")
    max_retries = 10
    initialized = False
    for attempt in range(max_retries):
        if orchestrator.initialize_shards():
            initialized = True
            break
        logger.warning(f"Worker shards not ready yet (attempt {attempt + 1}/{max_retries}). Retrying in 1s...")
        await asyncio.sleep(1.0)

    if not initialized:
        logger.error("Failed to initialize all worker shards. Aborting coordinator.")
        orchestrator.close()
        return

    logger.info("All worker shards initialized successfully! Starting coordinator server...")
    server = await serve_coordinator_async(
        cluster_config=cluster_config,
        orchestrator=orchestrator,
        host=cluster_config.coordinator_host,
        port=cluster_config.coordinator_port,
    )

    try:
        await server.wait_for_termination()
    except (asyncio.CancelledError, KeyboardInterrupt):
        await server.stop(0)
        orchestrator.close()


def main():
    parser = argparse.ArgumentParser(description="Distributed LLM Inference Engine Cluster Launcher")
    parser.add_argument("--world-size", type=int, default=2, help="Number of tensor parallel worker nodes")
    parser.add_argument("--base-port", type=int, default=50051, help="Starting port for worker nodes")
    parser.add_argument("--coord-port", type=int, default=50050, help="Coordinator port")
    parser.add_argument("--layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-kv-heads", type=int, default=2, help="Number of KV heads (GQA)")
    args = parser.parse_args()

    world_size = args.world_size
    worker_endpoints = [f"127.0.0.1:{args.base_port + i}" for i in range(world_size)]

    # Model configuration tailored for commodity hardware nodes
    model_config = ModelConfig(
        vocab_size=1000,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        intermediate_dim=int(args.hidden_dim * 2.5) // world_size * world_size,
        max_seq_len=1024,
    )

    cluster_config = ClusterConfig(
        coordinator_host="127.0.0.1",
        coordinator_port=args.coord_port,
        worker_endpoints=worker_endpoints,
        model=model_config,
        kv_cache=KVCacheConfig(block_size=16, num_blocks=128),
        scheduler=SchedulerConfig(max_batch_size=8),
    )

    logger.info(
        f"Launching cluster with {world_size} workers and Coordinator on port {args.coord_port}..."
    )

    # Launch worker processes
    processes: List[multiprocessing.Process] = []
    for rank in range(world_size):
        port = args.base_port + rank
        p = multiprocessing.Process(
            target=run_worker_process,
            args=(rank, world_size, "127.0.0.1", port),
            daemon=True,
        )
        p.start()
        processes.append(p)

    time.sleep(1.0)  # brief pause for workers to bind ports

    try:
        asyncio.run(run_coordinator_async(cluster_config))
    except KeyboardInterrupt:
        logger.info("Shutting down cluster...")
    finally:
        for p in processes:
            p.terminate()
            p.join(timeout=1.0)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
