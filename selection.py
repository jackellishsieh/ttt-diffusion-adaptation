"""
This file contains methods for selecting a preferred image, given their computed scores.
"""

import numpy as np
import numpy.typing as npt

def select(scores: npt.NDArray[np.floating], strategy: str = "argmax", **kwargs) -> int:
    """
    Select an index from scores using the specified strategy.
    
    Args:
        scores: Array of scores
        strategy: Selection strategy ("argmax", "softmax_sample")
        **kwargs: Additional arguments (e.g., temperature for softmax_sample)
    
    Returns:
        Selected index
    """
    if strategy == "argmax":
        return _argmax(scores)
    elif strategy == "softmax_sample":
        temp = float(kwargs.get("temperature", 1.0))
        return _softmax_sample(scores, temperature=temp)
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")


def _argmax(scores: npt.NDArray[np.floating]) -> int:
    """
    Determinsitically select the index with the maximum score.
    """
    if len(scores) == 0:
        raise ValueError("Empty scores")
    # np.argmax picks the first max in case of ties
    return int(np.argmax(scores))


def _softmax_sample(scores: npt.NDArray[np.floating], temperature: float = 1.0) -> int:
    """
    Sample an index according to Placket-Luce model: i.e., with softmax probabilities with temperature scaling.
    """
    if temperature <= 0:
        return _argmax(scores)
    
    # Compute softmax probabilities
    scaled_scores = scores / temperature
    exps = np.exp(scaled_scores - np.max(scaled_scores))        # Subtract max for numerical stability
    total = np.sum(exps)

    probs = exps / total
    return int(np.random.choice(len(scores), p=probs))


# Test stub
if __name__ == "__main__":
    scores = np.array([0.1, 0.9, 0.4])
    
    print("argmax:", select(scores, "argmax"))
    print("softmax sample:", select(scores, "softmax_sample", temperature=0.5))