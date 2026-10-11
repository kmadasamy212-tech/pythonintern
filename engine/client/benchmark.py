"""
Benchmark Suite for Distributed LLM Inference Engine.
Measures Time-To-First-Token (TTFT), Inter-Token Latency (ITL), and continuous batching throughput.
"""

from concurrent import futures
import time
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table

from engine.client.grpc_client import DistributedLLMClient
import proto_gen.distributed_inference_pb2 as pb2

console = Console()


def run_benchmark(
    target: str = "127.0.0.1:50050",
    num_requests: int = 8,
    max_tokens: int = 32,
    concurrency: int = 4,
) -> Dict[str, Any]:
    """Runs a distributed load benchmark against the inference cluster."""
    console.print(
        f"\n[bold green]=== Starting Distributed Engine Benchmark ===[/bold green]\n"
        f"Target: {target} | Requests: {num_requests} | Max Tokens: {max_tokens} | Concurrency: {concurrency}\n"
    )

    client = DistributedLLMClient(target=target)

    # First check cluster health
    try:
        status = client.get_cluster_status()
        console.print(
            f"[bold cyan]Cluster Status:[/bold cyan] {status.healthy_workers}/{status.num_workers} Workers Healthy\n"
        )
    except Exception as e:
        console.print(f"[bold red]Cluster unreachable:[/bold red] {e}")
        return {}

    test_prompts = [
        "Distributed systems enable scale across commodity nodes.",
        "Tensor parallelism partitions model weight matrices.",
        "Continuous batching eliminates head-of-line blocking in LLMs.",
        "Paged attention organizes KV caches into virtual blocks.",
        "Decentralized networks allow low-tier compute to participate.",
        "Asynchronous iteration scheduling boosts generation throughput.",
        "Row parallel layers execute All-Reduce collective sums.",
        "Column parallel layers partition output feature dimensions.",
    ]

    latencies = []
    ttfts = []
    total_tokens = 0
    start_total_time = time.perf_counter()

    def send_one(req_idx: int):
        prompt = test_prompts[req_idx % len(test_prompts)]
        t0 = time.perf_counter()
        req_ttft = None
        req_tokens = 0

        try:
            for resp in client.generate_stream(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.0,  # deterministic greedy
                request_id=f"bench_req_{req_idx}",
            ):
                if req_ttft is None:
                    req_ttft = resp.ttft_ms
                req_tokens += 1
            total_req_time = (time.perf_counter() - t0) * 1000.0
            return True, req_ttft or total_req_time, total_req_time, req_tokens
        except Exception as err:
            console.print(f"[red]Request {req_idx} failed: {err}[/red]")
            return False, 0.0, 0.0, 0

    with futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(send_one, range(num_requests)))

    total_wall_time_sec = time.perf_counter() - start_total_time

    successful_requests = 0
    for ok, ttft, lat, tok_cnt in results:
        if ok:
            successful_requests += 1
            ttfts.append(ttft)
            latencies.append(lat)
            total_tokens += tok_cnt

    avg_ttft = sum(ttfts) / max(1, len(ttfts))
    avg_latency = sum(latencies) / max(1, len(latencies))
    tokens_per_sec = total_tokens / max(0.001, total_wall_time_sec)
    itl_ms = (avg_latency - avg_ttft) / max(1, (max_tokens - 1)) if max_tokens > 1 else 0.0

    table = Table(title="Distributed LLM Inference Engine - Performance Benchmark")
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="magenta")

    table.add_row("Total Requests", str(num_requests))
    table.add_row("Successful Requests", str(successful_requests))
    table.add_row("Total Generated Tokens", str(total_tokens))
    table.add_row("Wall Time", f"{total_wall_time_sec:.2f} s")
    table.add_row("Throughput", f"[bold green]{tokens_per_sec:.2f} tokens/s[/bold green]")
    table.add_row("Mean TTFT (Time to First Token)", f"{avg_ttft:.2f} ms")
    table.add_row("Mean ITL (Inter-Token Latency)", f"{itl_ms:.2f} ms")
    table.add_row("Mean Total Request Latency", f"{avg_latency:.2f} ms")

    console.print(table)
    client.close()

    return {
        "throughput_tokens_sec": tokens_per_sec,
        "mean_ttft_ms": avg_ttft,
        "mean_itl_ms": itl_ms,
        "total_tokens": total_tokens,
        "wall_time_sec": total_wall_time_sec,
    }


if __name__ == "__main__":
    run_benchmark()
