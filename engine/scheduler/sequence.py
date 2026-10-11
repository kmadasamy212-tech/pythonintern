"""
Sequence and request state definitions for continuous batching.
"""

from enum import Enum
from typing import List, Optional, Any
import time
import asyncio


class SequenceStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED_STOP = "FINISHED_STOP"
    FINISHED_LENGTH = "FINISHED_LENGTH"
    PREEMPTED = "PREEMPTED"


class Sequence:
    """Represents a single generative sequence/request in the continuous batching pipeline."""

    def __init__(
        self,
        seq_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        stop_token_ids: Optional[List[int]] = None,
        stop_sequences: Optional[List[str]] = None,
    ):
        self.seq_id = seq_id
        self.prompt = prompt
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.status = SequenceStatus.WAITING

        # Sampling hyperparameters
        self.max_new_tokens = max_new_tokens
        self.temperature = max(1e-5, temperature)
        self.top_p = top_p
        self.top_k = top_k
        self.stop_token_ids = set(stop_token_ids or [0, 2])  # standard EOS/PAD
        self.stop_sequences = stop_sequences or []

        # Performance timestamps
        self.arrival_time = time.perf_counter()
        self.first_token_time: Optional[float] = None
        self.finish_time: Optional[float] = None

        # Async queue for real-time streaming to gRPC client
        self.stream_queue: Optional[asyncio.Queue] = None

    @property
    def total_length(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def current_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status in (SequenceStatus.FINISHED_STOP, SequenceStatus.FINISHED_LENGTH)

    def append_token(self, token_id: int):
        """Appends a newly sampled token and checks stopping criteria."""
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

        self.output_token_ids.append(token_id)

        if token_id in self.stop_token_ids:
            self.status = SequenceStatus.FINISHED_STOP
            self.finish_time = time.perf_counter()
        elif len(self.output_token_ids) >= self.max_new_tokens:
            self.status = SequenceStatus.FINISHED_LENGTH
            self.finish_time = time.perf_counter()
