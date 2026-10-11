"""
Interactive Streaming Chat CLI for Distributed LLM Inference Engine.
Connects to the Coordinator and streams token generation with real-time stats.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.panel import Panel
from engine.client.grpc_client import DistributedLLMClient

console = Console()


def interactive_chat(target: str = "127.0.0.1:50050"):
    console.print(
        Panel.fit(
            "[bold cyan]Distributed LLM Inference Engine - Interactive Client[/bold cyan]\n"
            f"[dim]Connected to Coordinator at {target}[/dim]\n"
            "[yellow]Type your prompt and press Enter. Type 'exit' to quit.[/yellow]"
        )
    )

    client = DistributedLLMClient(target=target)

    # Check status
    try:
        status = client.get_cluster_status()
        console.print(
            f"[green]Cluster Online:[/green] {status.healthy_workers}/{status.num_workers} active shards."
        )
    except Exception as e:
        console.print(f"[red]Error connecting to coordinator:[/red] {e}")
        return

    while True:
        try:
            prompt = console.input("\n[bold green]User > [/bold green]")
            if not prompt.strip():
                continue
            if prompt.strip().lower() in ("exit", "quit"):
                console.print("[dim]Goodbye![/dim]")
                break

            console.print("[bold blue]Assistant > [/bold blue]", end="")

            start_t = time.perf_counter()
            token_count = 0
            ttft = 0.0

            for resp in client.generate_stream(prompt=prompt, max_new_tokens=32, temperature=0.7):
                if token_count == 0:
                    ttft = resp.ttft_ms
                console.print(resp.text_delta, end="")
                sys.stdout.flush()
                token_count += 1

            total_lat = (time.perf_counter() - start_t) * 1000.0
            tps = token_count / max(0.001, (total_lat / 1000.0))

            console.print(
                f"\n[dim](Generated {token_count} tokens in {total_lat:.1f}ms | "
                f"TTFT: {ttft:.1f}ms | {tps:.1f} tokens/s)[/dim]"
            )

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"\n[red]Inference Error:[/red] {e}")

    client.close()


if __name__ == "__main__":
    interactive_chat()
