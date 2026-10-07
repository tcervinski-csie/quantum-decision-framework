"""Feedforward network (MLP) for Stage 2 classification, trained by backpropagation.

Architecture: INPUT_DIM -> hidden (ReLU) -> OUTPUT_DIM, softmax over the 5 reachable
(advantage, algorithm) pairs. One hidden layer is deliberate: the target function has
5 regions and two threshold boundaries, so 16 units is already generous. Extra depth
would fit the sampler's noise, not the function.

This module imports torch. It is reached only when the neural backend is explicitly
requested, so the package keeps working with torch uninstalled.
"""

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from quantum_agent.decision_engine import (
    ProblemFeatures,
    QuantumAdvantage,
    QuantumAlgorithm,
)
from quantum_agent.neural.encoding import (
    INPUT_DIM,
    OUTPUT_DIM,
    decode_label,
    encode_features,
)

HIDDEN_DIM = 16

WEIGHTS_PATH = Path(__file__).parent / "weights" / "advantage_mlp.pt"


class AdvantageMLP(nn.Module):
    """11 -> 16 (ReLU) -> 5. Outputs raw logits; callers apply softmax."""

    def __init__(self, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_cached_model: Optional[AdvantageMLP] = None


def load_model(path: Optional[Path] = None, force_reload: bool = False) -> AdvantageMLP:
    """Load the trained checkpoint, caching it across calls."""
    global _cached_model
    if _cached_model is not None and not force_reload and path is None:
        return _cached_model

    checkpoint_path = path or WEIGHTS_PATH
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained checkpoint at {checkpoint_path}. "
            "Run: python -m training.train_advantage_mlp"
        )

    model = AdvantageMLP()
    # weights_only=True refuses to unpickle arbitrary objects; the checkpoint is a
    # plain tensor state_dict, and this is the default from torch 2.6 onward.
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    model.eval()  # no dropout/batchnorm here, but keeps inference intent explicit

    if path is None:
        _cached_model = model
    return model


def predict(
    features: ProblemFeatures,
) -> tuple[QuantumAdvantage, QuantumAlgorithm, float]:
    """Classify features with the network.

    Returns the same 3-tuple shape as the rules backend. The confidence is the
    softmax probability of the winning class — unlike the rules' per-type constant,
    it varies per problem and drops near the decision boundaries, which is the
    signal the hybrid fusion strategy needs.
    """
    model = load_model()
    x = torch.tensor([encode_features(features)], dtype=torch.float32)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    index = int(torch.argmax(probs).item())
    advantage, algorithm = decode_label(index)
    return (advantage, algorithm, float(probs[index].item()))
