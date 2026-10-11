"""
Script to execute the distributed load and throughput benchmark.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.client.benchmark import run_benchmark

if __name__ == "__main__":
    run_benchmark(
        target="127.0.0.1:50050",
        num_requests=10,
        max_tokens=30,
        concurrency=4,
    )
