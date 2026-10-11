"""
Distributed Collective Communication primitives implemented over gRPC.
Enables All-Reduce, All-Gather, and Broadcast across decentralized commodity worker nodes.
"""

from concurrent import futures
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from engine.tensor_parallel.buffer_codec import serialize_tensor, deserialize_tensor
import proto_gen.distributed_inference_pb2 as pb2
import proto_gen.distributed_inference_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


class DistributedCommunicator:
    """
    Manages collective tensor communication over gRPC for a specific worker node.
    Supports decentralized All-Reduce and direct peer exchanges over TCP/IP.
    """

    def __init__(self, rank: int, world_size: int, peer_addresses: List[str] = None):
        self.rank = rank
        self.world_size = world_size
        self.peer_addresses = peer_addresses or []
        self._stubs: Dict[int, pb2_grpc.WorkerServiceStub] = {}
        self._channels = {}
        self._in_memory_peers: Dict[int, "DistributedCommunicator"] = {}
        self._executor = futures.ThreadPoolExecutor(max_workers=max(4, world_size * 2))

        # Inbound payload buffers: key = (step_tag, 0) -> dict(sender_rank -> tensor)
        self._inbound_lock = threading.Lock()
        self._inbound_events: Dict[Tuple[str, int], threading.Event] = {}
        self._inbound_payloads: Dict[Tuple[str, int], Dict[int, object]] = {}

    def register_peer_stub(self, peer_rank: int, stub: pb2_grpc.WorkerServiceStub):
        """Registers a gRPC WorkerServiceStub for a peer."""
        self._stubs[peer_rank] = stub

    def register_in_memory_peer(self, peer_rank: int, peer_comm: "DistributedCommunicator"):
        """Registers a direct in-memory peer communicator for local multi-rank testing."""
        self._in_memory_peers[peer_rank] = peer_comm

    def handle_incoming_allreduce(self, step_tag: str, sender_rank: int, tensor):
        """
        Callback invoked by the gRPC WorkerService or in-memory peer when a peer
        sends an AllReduce payload.
        """
        key = (step_tag, 0)
        with self._inbound_lock:
            if key not in self._inbound_payloads:
                self._inbound_payloads[key] = {}
            self._inbound_payloads[key][sender_rank] = tensor

            # Check if event is registered
            if key in self._inbound_events:
                # If we received from all expected peers
                if len(self._inbound_payloads[key]) >= (self.world_size - 1):
                    self._inbound_events[key].set()

    def all_reduce_sum(self, tensor, step_tag: str, timeout_sec: float = 15.0):
        """
        Synchronously performs All-Reduce (SUM) across all world_size nodes.
        If world_size == 1, returns tensor unchanged.
        For world_size > 1, exchanges partial products with peers via gRPC.
        """
        if self.world_size <= 1:
            return tensor

        key = (step_tag, 0)

        # Prepare event to wait for incoming peers
        with self._inbound_lock:
            if key not in self._inbound_payloads:
                self._inbound_payloads[key] = {}
            event = threading.Event()
            self._inbound_events[key] = event

            # If already satisfied by earlier incoming arrivals
            if len(self._inbound_payloads[key]) >= (self.world_size - 1):
                event.set()

        # Send our tensor slice to all peers concurrently
        raw_bytes, shape, dtype_str = serialize_tensor(tensor)
        self._broadcast_to_peers(step_tag, raw_bytes, shape, dtype_str, tensor, timeout_sec)

        # Wait for all peers to send their tensors
        signaled = event.wait(timeout=timeout_sec)
        if not signaled:
            with self._inbound_lock:
                received = list(self._inbound_payloads.get(key, {}).keys())
            raise TimeoutError(
                f"[Rank {self.rank}] All-Reduce timeout on tag '{step_tag}'. "
                f"Expected {self.world_size - 1} peers, received from ranks: {received}"
            )

        # Aggregate (SUM) all received tensors into our local tensor
        with self._inbound_lock:
            peer_tensors = self._inbound_payloads.pop(key, {})
            self._inbound_events.pop(key, None)

        # Sum local tensor + peer tensors
        res = tensor.clone() if HAS_TORCH and isinstance(tensor, torch.Tensor) else tensor.copy()
        for p_rank, p_tensor in peer_tensors.items():
            res += p_tensor

        return res

    def _broadcast_to_peers(
        self,
        step_tag: str,
        raw_bytes: bytes,
        shape: Tuple[int, ...],
        dtype_str: str,
        local_tensor,
        timeout_sec: float,
    ):
        """Sends tensor payload to all peers asynchronously."""
        # 1. In-memory peers
        for p_rank, p_comm in self._in_memory_peers.items():
            if p_rank != self.rank:
                clone = local_tensor.clone() if HAS_TORCH and isinstance(local_tensor, torch.Tensor) else local_tensor.copy()
                p_comm.handle_incoming_allreduce(step_tag, self.rank, clone)

        # 2. gRPC peers
        if self._stubs:
            payload = pb2.AllReducePayload(
                step_tag=step_tag,
                sender_rank=self.rank,
                shape=list(shape),
                dtype=dtype_str,
                raw_tensor_data=raw_bytes,
            )

            def send(p_rank, stub):
                try:
                    stub.ExchangeAllReduce(payload, timeout=timeout_sec)
                except Exception as e:
                    logger.error(f"[Rank {self.rank}] Error sending AllReduce to peer {p_rank}: {e}")

            for p_rank, stub in self._stubs.items():
                if p_rank != self.rank:
                    self._executor.submit(send, p_rank, stub)

    def cleanup(self):
        """Closes gRPC channels and shuts down executor."""
        for ch in self._channels.values():
            try:
                ch.close()
            except Exception:
                pass
        self._channels.clear()
        self._stubs.clear()
        self._in_memory_peers.clear()
        self._executor.shutdown(wait=False)
