# Decentralized Distributed LLM Inference Engine

A custom, distributed large language model (LLM) inference engine built with **PyTorch** and **gRPC** that shards model weights across multiple low-tier commodity hardware nodes using **Tensor Parallelism** and **Continuous Batching**. Includes official entry points for **Vercel Serverless Functions**, Docker, and standalone execution.

---

## 🚀 Key Architectural Innovations

### 1. Decentralized Tensor Parallelism (TP) over gRPC
Traditional tensor parallelism frameworks (e.g., Megatron-LM, vLLM) depend on tightly coupled NVLink or InfiniBand interconnects within high-end server nodes. This engine enables **cross-node tensor sharding over standard TCP/IP networks**:
* **Column-Parallel Linear Layers** (`ColumnParallelLinear`): Weight matrices ($W \in \mathbb{R}^{d_{in} \times d_{out}}$) are partitioned column-wise across $N$ commodity nodes ($W_i \in \mathbb{R}^{d_{in} \times \frac{d_{out}}{N}}$). Forward projections ($Y_i = X W_i$) run concurrently without communication.
* **Row-Parallel Linear Layers** (`RowParallelLinear`): Weight matrices are partitioned row-wise ($W_i \in \mathbb{R}^{\frac{d_{in}}{N} \times d_{out}}$). Each worker calculates a local partial product ($Y_i = X_i W_i$), followed by an asynchronous **gRPC All-Reduce (SUM)** to reconstruct the output activation:
  $$\mathbf{Y} = \sum_{i=0}^{N-1} \mathbf{X}_i \mathbf{W}_i$$
* **Zero-Overhead Raw Buffer Codec** (`engine/tensor_parallel/buffer_codec.py`): Replaces JSON / generic Protobuf object serialization with contiguous raw-byte memory transfer and zero-copy pointer views (`torch.frombuffer`), minimizing CPU deserialization overhead on commodity nodes.

### 2. Paged KV Cache Memory Manager
Commodity nodes (e.g., 4GB–16GB RAM) suffer heavily from memory fragmentation when pre-allocating contiguous buffers for maximum sequence lengths.
* Physical Key-Value memory is divided into fixed-size **Blocks** (e.g., `block_size = 16`).
* Dynamic **Block Tables** map logical token positions $t$ to physical block slots:
  $$\text{Block ID} = \lfloor t / \text{block\_size} \rfloor, \quad \text{Offset} = t \pmod{\text{block\_size}}$$
* Eliminates internal and external memory fragmentation, freeing blocks immediately upon sequence completion.

### 3. Iteration-Level Continuous Batching (In-Flight Batching)
Traditional static batching suffers from severe head-of-line blocking when prompt or generation lengths vary. The engine's **Continuous Batcher** (`engine/scheduler/continuous_batcher.py`) dynamically creates batches at every execution step:
* **Decode Phase**: Advances ongoing in-flight generations token-by-token.
* **Prefill Phase**: Immediately admits waiting requests into spare batch and memory capacity without waiting for existing decodes to complete.
* **Eviction & Recycling**: Recycles memory blocks and terminates requests upon encountering EOS tokens or reaching token limits.

### 4. Vercel Serverless & REST Support
* **Root Entrypoint** (`main.py` / `app.py`): Full ASGI FastAPI application exposing REST and Server-Sent Events (SSE) streaming endpoints.
* **Vercel Serverless Entrypoint** (`api/index.py`): Native serverless handler exported for `@vercel/python`.
* **Zero-Configuration Routing** (`vercel.json`): Directs incoming web traffic to the serverless entry point.

---

## 📂 Project Directory Structure

```
├── main.py                              # Root Python entry-point (FastAPI / ASGI / Vercel)
├── app.py                               # Root alias entry-point
├── vercel.json                          # Vercel deployment routing and build specification
├── api/
│   └── index.py                         # Vercel Serverless Function entry-point
├── protos/
│   └── distributed_inference.proto      # gRPC service and collective definitions
├── proto_gen/                           # Compiled Python gRPC stubs & descriptors
├── engine/
│   ├── config.py                        # Model, Shard, KV Cache & Scheduler configurations
│   ├── tensor_parallel/
│   │   ├── comm.py                      # Distributed gRPC collectives (All-Reduce, Broadcast)
│   │   ├── layers.py                    # ColumnParallelLinear, RowParallelLinear, RMSNorm
│   │   ├── sharding.py                  # Weight partitioning and state-dict sharding
│   │   └── buffer_codec.py              # Zero-copy raw tensor serialization
│   ├── model/
│   │   ├── llama.py                     # Sharded LLaMA / Mistral Transformer architecture
│   │   ├── attention.py                 # Sharded GQA Attention with RoPE and Paged KV Cache
│   │   ├── mlp.py                       # Sharded SwiGLU Feed-Forward Network
│   │   └── kv_cache.py                  # Physical block pool and block table manager
│   ├── scheduler/
│   │   ├── sequence.py                  # Request states (WAITING, RUNNING, FINISHED)
│   │   ├── sampler.py                   # Greedy, Temperature, Top-K, and Top-P sampling
│   │   ├── memory_manager.py            # Centralized KV block table allocator
│   │   └── continuous_batcher.py        # Iteration-level continuous batching scheduler
│   ├── worker/
│   │   ├── execution_engine.py          # Local model shard execution and KV cache updates
│   │   └── worker_node.py               # gRPC Worker Service daemon
│   ├── coordinator/
│   │   ├── cluster_orchestrator.py      # Multi-worker step dispatch & collective synchronization
│   │   └── coordinator.py               # Client streaming gRPC server & scheduler event loop
│   └── client/
│       ├── grpc_client.py               # High-throughput streaming gRPC client SDK
│       └── benchmark.py                 # Multi-request throughput & latency benchmark suite
├── scripts/
│   ├── launch_cluster.py                # Launcher for multi-node worker & coordinator cluster
│   ├── demo_chat.py                     # Interactive streaming terminal chat
│   └── run_benchmark.py                 # Distributed load benchmark script
├── tests/
│   ├── test_api_entrypoint.py           # Tests for root main.py and api/index.py Vercel entrypoint
│   ├── test_tensor_parallel.py          # Tests for Column/Row linear layers & buffer codec
│   ├── test_continuous_batching.py      # Tests for iteration scheduling & prefill interleaving
│   ├── test_grpc_collectives.py         # Tests for 4-node All-Reduce collective communications
│   ├── test_kv_cache.py                 # Tests for physical block pools & block tables
│   └── test_end_to_end_inference.py     # Live multi-worker distributed streaming test
├── requirements.txt                     # Lightweight Vercel serverless requirements (<100MB)
├── requirements-cluster.txt             # Full PyTorch & worker cluster requirements
├── .vercelignore                        # Excludes test caches, scripts, and temporary files
└── README.md
```

---

## 🛠️ Quickstart Guide

### 1. Install Dependencies
* **For Vercel / Web API Gateway (<100MB bundle size)**:
  ```powershell
  pip install -r requirements.txt
  ```
* **For Distributed Commodity Worker Nodes & Full Cluster**:
  ```powershell
  pip install -r requirements-cluster.txt
  ```

### 2. Run All Tests
Verify all 17 unit and integration tests across tensor parallelism, collectives, KV caching, continuous batching, and API entrypoints:
```powershell
python -m pytest tests/ -v
```

### 3. Launch Web API / Vercel Local Server
Start the REST/SSE web server locally:
```powershell
python main.py
```
Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser to inspect interactive Swagger API documentation.

### 4. Deploying to Vercel
1. Ensure the repository contains `main.py`, `api/index.py`, `vercel.json`, and `.vercelignore`.
2. Deploy via the Vercel CLI or Git push:
   ```bash
   vercel
   ```
3. Vercel will automatically discover `api/index.py` using `@vercel/python`, serve the FastAPI app, and handle requests to `/`, `/health`, `/generate`, and `/generate/stream`.

### 5. Launch Distributed Cluster (gRPC)
To spin up a distributed inference cluster across 2 worker nodes (ports `50051`, `50052`) and the Coordinator (port `50050`):
```powershell
python scripts/launch_cluster.py --world-size 2 --base-port 50051 --coord-port 50050
```

### 6. Interactive Streaming Client & Benchmarks
```powershell
python scripts/demo_chat.py
python scripts/run_benchmark.py
```
