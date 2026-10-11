"""
High-performance token sampler for continuous batching.
Supports greedy sampling, temperature scaling, Top-K, and Top-P (nucleus) sampling.
"""

from typing import List
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from engine.scheduler.sequence import Sequence


class Sampler:
    """Samples next token IDs given logits from model forward step."""

    @staticmethod
    def sample(logits: "torch.Tensor", sequences: List[Sequence]) -> List[int]:
        """
        logits: [batch_size, vocab_size]
        sequences: list of Sequence objects of length batch_size
        returns: list of next token IDs
        """
        if HAS_TORCH and isinstance(logits, torch.Tensor):
            return Sampler._sample_torch(logits, sequences)
        else:
            return Sampler._sample_numpy(logits, sequences)

    @staticmethod
    def _sample_torch(logits: "torch.Tensor", sequences: List[Sequence]) -> List[int]:
        batch_size, vocab_size = logits.shape
        next_tokens = []

        for i, seq in enumerate(sequences):
            seq_logits = logits[i].clone()

            # Greedy decoding
            if seq.temperature <= 1e-4:
                token_id = int(torch.argmax(seq_logits).item())
                next_tokens.append(token_id)
                continue

            # Temperature scaling
            seq_logits = seq_logits / seq.temperature

            # Top-K filtering
            if seq.top_k > 0 and seq.top_k < vocab_size:
                top_k_val, _ = torch.topk(seq_logits, seq.top_k)
                min_top_k = top_k_val[-1]
                seq_logits[seq_logits < min_top_k] = float("-inf")

            # Top-P (Nucleus) filtering
            if seq.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(seq_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > seq.top_p
                # Shift right to keep at least first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                seq_logits[indices_to_remove] = float("-inf")

            # Sample from categorical distribution
            probs = F.softmax(seq_logits, dim=-1)
            token_id = int(torch.multinomial(probs, num_samples=1).item())
            next_tokens.append(token_id)

        return next_tokens

    @staticmethod
    def _sample_numpy(logits: np.ndarray, sequences: List[Sequence]) -> List[int]:
        next_tokens = []
        for i, seq in enumerate(sequences):
            seq_logits = np.copy(logits[i])
            if seq.temperature <= 1e-4:
                next_tokens.append(int(np.argmax(seq_logits)))
                continue

            seq_logits = seq_logits / seq.temperature
            # Softmax
            exp_logits = np.exp(seq_logits - np.max(seq_logits))
            probs = exp_logits / np.sum(exp_logits)

            # Sample
            token_id = int(np.random.choice(len(probs), p=probs))
            next_tokens.append(token_id)
        return next_tokens
