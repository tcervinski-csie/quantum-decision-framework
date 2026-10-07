"""Quantum Decision Framework: classify problems and gate hardware feasibility.

Three-stage pipeline:
1. Feature Extraction — analyze problem structure
2. Quantum Advantage Classification — clear / potential / no current / classical preferred
3. Hardware Feasibility Gate — check against real backend constraints
"""

from dataclasses import dataclass, field
from enum import Enum
from math import log2, ceil
from typing import Optional


class ProblemType(Enum):
    UNSTRUCTURED_SEARCH = "unstructured_search"
    COMBINATORIAL_OPTIMIZATION = "combinatorial_optimization"
    QUANTUM_SIMULATION = "quantum_simulation"
    LINEAR_ALGEBRA = "linear_algebra"
    CRYPTOGRAPHIC = "cryptographic"
    MACHINE_LEARNING = "machine_learning"
    OTHER = "other"


class QuantumAdvantage(Enum):
    CLEAR = "clear_advantage"
    POTENTIAL = "potential_advantage"
    NO_CURRENT = "no_current_advantage"
    CLASSICAL_PREFERRED = "classical_preferred"


class QuantumAlgorithm(Enum):
    GROVER = "grover"
    QAOA = "qaoa"
    VQE = "vqe"
    NONE = "none"


class ExecutionTarget(Enum):
    CLASSICAL = "classical"
    QUANTUM_SIMULATE = "quantum_simulate"
    QUANTUM_HARDWARE = "quantum_hardware"


@dataclass
class ProblemFeatures:
    """Extracted features from a problem description."""
    problem_type: ProblemType
    search_space_size: int
    has_oracle: bool = False
    has_structure: bool = False
    num_variables: int = 0
    requires_fault_tolerance: bool = False


@dataclass
class HardwareConstraints:
    """Constraints of the target quantum backend.

    `backend_name` selects how circuit depth is obtained. Left as None, depth comes
    from the analytic estimate in `estimate_resources`. Set to a fake-backend name
    (e.g. "fake_brisbane"), the circuit is compiled for that device's real coupling
    map and basis gates and the measured depth is used instead. The analytic
    estimate understates compiled depth by 20-160x for Grover and the gap widens
    with problem size, so the two paths can route very differently.
    """
    max_qubits: int = 127          # IBM Eagle
    max_circuit_depth: int = 100   # practical NISQ limit
    connectivity: str = "heavy_hex" # IBM topology
    backend_name: Optional[str] = None  # None = analytic depth; else measure it


@dataclass
class Decision:
    """Output of the decision framework."""
    advantage: QuantumAdvantage
    algorithm: QuantumAlgorithm
    target: ExecutionTarget
    estimated_qubits: int
    estimated_depth: int
    confidence: float  # 0.0 to 1.0
    reasoning: str
    hardware_feasible: bool = True


# ---------------------------------------------------------------------------
# Stage 1: Feature Extraction
# ---------------------------------------------------------------------------

def extract_features(
    problem_type: str,
    search_space_size: int,
    has_oracle: bool = False,
    has_structure: bool = False,
    num_variables: int = 0,
) -> ProblemFeatures:
    """Parse raw problem attributes into a ProblemFeatures object."""
    try:
        ptype = ProblemType(problem_type)
    except ValueError:
        ptype = ProblemType.OTHER

    requires_ft = ptype in (ProblemType.CRYPTOGRAPHIC, ProblemType.LINEAR_ALGEBRA)

    return ProblemFeatures(
        problem_type=ptype,
        search_space_size=search_space_size,
        has_oracle=has_oracle,
        has_structure=has_structure,
        num_variables=num_variables,
        requires_fault_tolerance=requires_ft,
    )


# ---------------------------------------------------------------------------
# Stage 2: Quantum Advantage Classification
# ---------------------------------------------------------------------------

# Minimum search space for Grover's to beat classical brute force.
# Below this, the overhead of state preparation negates the quadratic speedup.
GROVER_MIN_SEARCH_SPACE = 64

# Maps problem types to (advantage, algorithm, confidence) when quantum applies.
_CLASSIFICATION_TABLE: dict[ProblemType, tuple[QuantumAdvantage, QuantumAlgorithm, float]] = {
    ProblemType.UNSTRUCTURED_SEARCH: (QuantumAdvantage.CLEAR, QuantumAlgorithm.GROVER, 0.85),
    ProblemType.COMBINATORIAL_OPTIMIZATION: (QuantumAdvantage.POTENTIAL, QuantumAlgorithm.QAOA, 0.55),
    ProblemType.QUANTUM_SIMULATION: (QuantumAdvantage.CLEAR, QuantumAlgorithm.VQE, 0.90),
    ProblemType.CRYPTOGRAPHIC: (QuantumAdvantage.NO_CURRENT, QuantumAlgorithm.NONE, 0.95),
    ProblemType.LINEAR_ALGEBRA: (QuantumAdvantage.NO_CURRENT, QuantumAlgorithm.NONE, 0.80),
    ProblemType.MACHINE_LEARNING: (QuantumAdvantage.CLASSICAL_PREFERRED, QuantumAlgorithm.NONE, 0.70),
}


def classify_advantage(
    features: ProblemFeatures,
    backend: str = "rules",
) -> tuple[QuantumAdvantage, QuantumAlgorithm, float]:
    """Classify quantum advantage based on problem features.

    Returns (advantage_level, recommended_algorithm, confidence).

    backend:
        "rules"  — the deterministic table below (default; unchanged behaviour).
        "neural" — a distilled feedforward network. Requires the optional torch
                   dependency and a trained checkpoint; see training/.
    """
    if backend == "neural":
        # Imported lazily so torch stays optional for the default path.
        from quantum_agent.neural.model import predict

        return predict(features)
    if backend != "rules":
        raise ValueError(f"Unknown classification backend: {backend!r}")

    # Unknown problem types default to classical.
    if features.problem_type == ProblemType.OTHER:
        return (QuantumAdvantage.CLASSICAL_PREFERRED, QuantumAlgorithm.NONE, 0.5)

    advantage, algorithm, confidence = _CLASSIFICATION_TABLE[features.problem_type]

    # Grover's needs an oracle and a large enough search space.
    if features.problem_type == ProblemType.UNSTRUCTURED_SEARCH:
        if not features.has_oracle:
            return (QuantumAdvantage.CLASSICAL_PREFERRED, QuantumAlgorithm.NONE, 0.75,)
        if features.search_space_size < GROVER_MIN_SEARCH_SPACE:
            return (QuantumAdvantage.CLASSICAL_PREFERRED, QuantumAlgorithm.NONE, 0.80)

    # Fault-tolerant algorithms can't run on current hardware.
    if features.requires_fault_tolerance:
        return (QuantumAdvantage.NO_CURRENT, QuantumAlgorithm.NONE, confidence)

    return (advantage, algorithm, confidence)


# ---------------------------------------------------------------------------
# Stage 3: Hardware Feasibility Gate
# ---------------------------------------------------------------------------

def estimate_resources(
    algorithm: QuantumAlgorithm,
    features: ProblemFeatures,
) -> tuple[int, int]:
    """Estimate qubits and circuit depth for a given algorithm + problem.

    Returns (estimated_qubits, estimated_depth).
    """
    if algorithm == QuantumAlgorithm.GROVER:
        n_qubits = max(1, ceil(log2(features.search_space_size)))
        # Grover needs ~sqrt(N) iterations, each iteration is O(n) depth.
        iterations = max(1, ceil(features.search_space_size ** 0.5))
        depth = iterations * n_qubits
        return (n_qubits, depth)

    if algorithm == QuantumAlgorithm.QAOA:
        n_qubits = features.num_variables if features.num_variables > 0 else max(1, ceil(log2(features.search_space_size)))
        # QAOA with p=1 round: ~2*n_qubits depth per layer.
        depth = 2 * n_qubits
        return (n_qubits, depth)

    if algorithm == QuantumAlgorithm.VQE:
        n_qubits = features.num_variables if features.num_variables > 0 else 4
        depth = 4 * n_qubits  # ansatz depth scales with qubit count
        return (n_qubits, depth)

    return (0, 0)


def check_feasibility(
    estimated_qubits: int,
    estimated_depth: int,
    constraints: Optional[HardwareConstraints] = None,
) -> tuple[bool, ExecutionTarget]:
    """Gate: can this circuit run on real hardware, or must we simulate?

    Returns (is_feasible_on_hardware, recommended_target).
    """
    if constraints is None:
        constraints = HardwareConstraints()

    if estimated_qubits == 0:
        return (False, ExecutionTarget.CLASSICAL)

    fits_hardware = (
        estimated_qubits <= constraints.max_qubits
        and estimated_depth <= constraints.max_circuit_depth
    )

    if fits_hardware:
        return (True, ExecutionTarget.QUANTUM_HARDWARE)

    # Too large for hardware but still worth simulating if qubits are manageable.
    # Aer can simulate ~30 qubits on a typical machine.
    if estimated_qubits <= 30:
        return (False, ExecutionTarget.QUANTUM_SIMULATE)

    return (False, ExecutionTarget.CLASSICAL)


# ---------------------------------------------------------------------------
# Public API: full pipeline
# ---------------------------------------------------------------------------

def decide(
    problem_type: str,
    search_space_size: int,
    has_oracle: bool = False,
    has_structure: bool = False,
    num_variables: int = 0,
    hardware: Optional[HardwareConstraints] = None,
    backend: str = "rules",
) -> Decision:
    """Run the full three-stage decision pipeline.

    This is the main entry point for the framework.

    `backend` selects the Stage 2 classifier only. Stages 1 and 3 are always
    deterministic — the feasibility gate checks hard hardware limits, so an
    approximation of it could admit a circuit the backend cannot run.
    """
    # Stage 1
    features = extract_features(
        problem_type=problem_type,
        search_space_size=search_space_size,
        has_oracle=has_oracle,
        has_structure=has_structure,
        num_variables=num_variables,
    )

    # Stage 2
    advantage, algorithm, confidence = classify_advantage(features, backend=backend)

    # If classical preferred or no current advantage, short-circuit.
    if algorithm == QuantumAlgorithm.NONE:
        return Decision(
            advantage=advantage,
            algorithm=algorithm,
            target=ExecutionTarget.CLASSICAL,
            estimated_qubits=0,
            estimated_depth=0,
            confidence=confidence,
            reasoning=_build_reasoning(features, advantage, algorithm, True),
            hardware_feasible=False,
        )

    # Stage 3
    est_qubits, est_depth = estimate_resources(algorithm, features)

    # When a backend is named, replace the analytic depth with the real compiled
    # depth for that device. Imported lazily to avoid a module-level import cycle
    # through code_generator.
    if hardware is not None and hardware.backend_name:
        from quantum_agent.transpilation import transpiled_depth

        est_depth = transpiled_depth(
            algorithm, est_qubits, est_depth, hardware.backend_name
        )

    hw_feasible, target = check_feasibility(est_qubits, est_depth, hardware)

    return Decision(
        advantage=advantage,
        algorithm=algorithm,
        target=target,
        estimated_qubits=est_qubits,
        estimated_depth=est_depth,
        confidence=confidence,
        reasoning=_build_reasoning(features, advantage, algorithm, hw_feasible),
        hardware_feasible=hw_feasible,
    )


def _build_reasoning(
    features: ProblemFeatures,
    advantage: QuantumAdvantage,
    algorithm: QuantumAlgorithm,
    hw_feasible: bool,
) -> str:
    """Generate a human-readable explanation of the decision."""
    parts = [
        f"Problem type: {features.problem_type.value}",
        f"Search space: {features.search_space_size}",
        f"Advantage: {advantage.value}",
        f"Algorithm: {algorithm.value}",
        f"Hardware feasible: {hw_feasible}",
    ]
    if features.requires_fault_tolerance:
        parts.append("Requires fault-tolerant QC — not available on NISQ hardware.")
    if not features.has_oracle and features.problem_type == ProblemType.UNSTRUCTURED_SEARCH:
        parts.append("No oracle available — Grover's algorithm not applicable.")
    return " | ".join(parts)
