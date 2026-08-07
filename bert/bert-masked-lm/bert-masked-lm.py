import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    mask_positions: np.ndarray,
    replace_probs: np.ndarray,
    random_tokens: np.ndarray,
    mask_token_id: int = 103
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: tuple of (np.ndarray masked_ids, np.ndarray labels) with masking applied
    """
    # YOUR CODE HERE
    # B, seq_len = token_ids
    labels = np.where(mask_positions, token_ids, -100)
    masked_id = token_ids.copy()
    # assert False, labels
    masked_id = np.where(
        mask_positions & (replace_probs < .9),
        random_tokens,
        masked_id
    )
    masked_id = np.where(
        mask_positions & (replace_probs < .8),
        np.full_like(labels, mask_token_id),
        masked_id
    )

    return masked_id, labels

class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token logits: hidden_states @ W + b
        """
        # YOUR CODE HERE
        return hidden_states @ self.W + self.b
        
