"""
High-Performance gRPC Client for Distributed LLM Inference Engine.
Provides both streaming and blocking generation interfaces with real-time token yield.
"""

from typing import Iterator, List, Optional
import grpc

import proto_gen.distributed_inference_pb2 as pb2
import proto_gen.distributed_inference_pb2_grpc as pb2_grpc


class DistributedLLMClient:
    """Client for connecting to the Distributed Inference Coordinator."""

    def __init__(self, target: str = "127.0.0.1:50050"):
        self.target = target
        self.channel = grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ],
        )
        self.stub = pb2_grpc.InferenceCoordinatorServiceStub(self.channel)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        request_id: Optional[str] = None,
    ) -> Iterator[pb2.GenerateResponse]:
        """Streams generated tokens from the continuous batching coordinator."""
        sampling = pb2.SamplingParamsProto(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )
        req = pb2.GenerateRequest(
            request_id=request_id or "",
            prompt=prompt,
            sampling_params=sampling,
        )
        return self.stub.GenerateStream(req)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """Convenience method that buffers the entire response and returns completed text."""
        tokens = []
        for resp in self.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ):
            tokens.append(resp.text_delta)
        return "".join(tokens)

    def get_cluster_status(self) -> pb2.ClusterStatusResponse:
        return self.stub.GetClusterStatus(pb2.ClusterStatusRequest())

    def close(self):
        self.channel.close()
