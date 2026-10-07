"""Feature encoding shared by training and inference.

Both the training script and the inference path MUST import this module. If the
two encode features differently — a different one-hot order, a different log base,
different scaling constants — the model scores well during training and behaves
randomly in production. Sharing one definition is the only defense against that
train/serve skew, so the constants below are fixed here rather than derived from
whatever data happens to be at hand.

Requires numpy only; torch is not imported here.
"""

from math import log1p, log2

from quantum_agent.decision_engine import (
    ProblemFeatures,
    ProblemType,
    QuantumAdvantage,
    QuantumAlgorithm,
)

# Fixed one-hot order for ProblemType. Appending a new member to the enum is safe;
# reordering existing members invalidates every trained checkpoint.
PROBLEM_TYPE_ORDER: tuple[ProblemType, ...] = tuple(ProblemType)

# Scaling constants. search_space_size spans 1 to ~10^6, so it is fed as log2 —
# raw magnitudes would dominate every gradient. log2 also matches the physics:
# qubit count IS log2(N), so the network receives the quantity the problem turns on.
LOG2_SEARCH_SPACE_SCALE = 20.0   # log2(10^6) ~= 19.9
LOG_NUM_VARIABLES_SCALE = 6.0    # log1p(200) ~= 5.3

INPUT_DIM = len(PROBLEM_TYPE_ORDER) + 4  # 7 one-hot + oracle + structure + 2 sizes

# Fixed label order. The 5 reachable (advantage, algorithm) pairs; see the grid
# sweep in the design notes — the other combinations are unreachable by construction.
CLASSES: tuple[tuple[QuantumAdvantage, QuantumAlgorithm], ...] = (
    (QuantumAdvantage.CLASSICAL_PREFERRED, QuantumAlgorithm.NONE),
    (QuantumAdvantage.CLEAR, QuantumAlgorithm.GROVER),
    (QuantumAdvantage.CLEAR, QuantumAlgorithm.VQE),
    (QuantumAdvantage.NO_CURRENT, QuantumAlgorithm.NONE),
    (QuantumAdvantage.POTENTIAL, QuantumAlgorithm.QAOA),
)
OUTPUT_DIM = len(CLASSES)

_CLASS_INDEX = {pair: i for i, pair in enumerate(CLASSES)}


def encode_features(features: ProblemFeatures) -> list[float]:
    """Encode ProblemFeatures into the fixed-length input vector."""
    vec = [0.0] * INPUT_DIM

    # One-hot problem type. An unknown type leaves the block all-zero rather than
    # raising, which keeps inference total on inputs the enum does not cover.
    try:
        vec[PROBLEM_TYPE_ORDER.index(features.problem_type)] = 1.0
    except ValueError:
        pass

    offset = len(PROBLEM_TYPE_ORDER)
    vec[offset] = 1.0 if features.has_oracle else 0.0
    vec[offset + 1] = 1.0 if features.has_structure else 0.0
    vec[offset + 2] = log2(max(1, features.search_space_size)) / LOG2_SEARCH_SPACE_SCALE
    vec[offset + 3] = log1p(max(0, features.num_variables)) / LOG_NUM_VARIABLES_SCALE
    return vec


def label_index(advantage: QuantumAdvantage, algorithm: QuantumAlgorithm) -> int:
    """Map a rules outcome to its class index (the distillation target)."""
    return _CLASS_INDEX[(advantage, algorithm)]


def decode_label(index: int) -> tuple[QuantumAdvantage, QuantumAlgorithm]:
    """Map a predicted class index back to (advantage, algorithm)."""
    return CLASSES[index]
