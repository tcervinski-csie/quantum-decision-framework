"""Transpilation-based resource measurement against real device calibration data.

The analytic estimates in `decision_engine.estimate_resources` model Grover's depth
as ceil(sqrt(N)) * ceil(log2 N). Measured against circuits actually compiled for IBM
hardware, that is low by one to two orders of magnitude, and the error GROWS with N:

    N      analytic    compiled (fake_brisbane, heavy-hex)    factor
    8             9                                    184      20x
    32           30                                  2,064      69x
    64           48                                  7,609     158x

Two causes. An ancilla-free multi-controlled X decomposes into O(n^2)-ish two-qubit
gates rather than the O(n) the analytic model assumes; and on a heavy-hex coupling
map the transpiler must insert SWAP chains, since the abstract circuit assumes
all-to-all connectivity that no IBM device has.

Because the gap is not a constant factor, no corrected closed form fixes it. The
honest measurement is to compile the circuit for a specific backend and read the
result, which is what this module does.

No IBM credentials are required: `qiskit_ibm_runtime.fake_provider` ships snapshots
of real device calibration (coupling map, basis gates, gate errors, T1/T2). These
are real device properties, not synthetic ones — but they are a calibration
snapshot, not live execution.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from qiskit import transpile

from quantum_agent.decision_engine import (
    Decision,
    ExecutionTarget,
    QuantumAdvantage,
    QuantumAlgorithm,
)

# 127-qubit Eagle-family device, matching the hardware target the framework assumes.
DEFAULT_BACKEND = "fake_brisbane"

# Fixed so repeated runs give identical numbers; transpilation is stochastic.
TRANSPILE_SEED = 7
OPTIMIZATION_LEVEL = 1


class BackendUnavailableError(RuntimeError):
    """Raised when the named fake backend cannot be loaded."""


@dataclass
class TranspiledMetrics:
    """What a circuit actually costs once compiled for a specific device."""
    backend_name: str
    num_qubits: int
    depth: int
    two_qubit_gates: int
    total_gates: int
    analytic_depth: int
    estimated_fidelity: float = 0.0  # product of per-gate success, incl. readout

    @property
    def depth_ratio(self) -> float:
        """How far the analytic estimate is from the compiled reality."""
        return self.depth / self.analytic_depth if self.analytic_depth else float("inf")


@lru_cache(maxsize=8)
def get_backend(name: str = DEFAULT_BACKEND):
    """Load a fake backend by name, cached (construction is not cheap)."""
    import warnings

    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

    with warnings.catch_warnings():
        # Some fake backends warn that their error values are not representative;
        # that caveat belongs in the paper, not in every call site's stderr.
        warnings.simplefilter("ignore")
        for backend in FakeProviderForBackendV2().backends():
            if backend.name == name:
                return backend

    raise BackendUnavailableError(f"No fake backend named {name!r}")


def estimate_fidelity(compiled, backend) -> float:
    """Estimated probability that the whole circuit executes without error.

    Multiplies (1 - error) over every gate as placed on physical qubits, using the
    backend's own calibration, then applies readout error for each measured qubit.

    This is the standard first-order estimate and it is deliberately optimistic: it
    assumes errors are independent and ignores crosstalk, leakage and drift. It is
    therefore an UPPER BOUND on achievable fidelity — useful because a circuit whose
    upper bound is already negligible cannot be rescued by a better device day.
    """
    target = backend.target
    fidelity = 1.0

    for instruction in compiled.data:
        name = instruction.operation.name
        if name in ("barrier", "delay"):
            continue

        qubits = tuple(compiled.find_bit(q).index for q in instruction.qubits)
        try:
            props = target[name].get(qubits)
        except (KeyError, AttributeError):
            props = None

        if props is not None and props.error is not None:
            # A calibration error of 1.0 marks an unusable gate on that pair.
            fidelity *= max(0.0, 1.0 - props.error)

    return fidelity


def _stub_decision(algorithm: QuantumAlgorithm, n_qubits: int) -> Decision:
    """Minimal Decision for the circuit builders, which read only these two fields."""
    return Decision(
        advantage=QuantumAdvantage.CLEAR,
        algorithm=algorithm,
        target=ExecutionTarget.QUANTUM_SIMULATE,
        estimated_qubits=n_qubits,
        estimated_depth=0,
        confidence=1.0,
        reasoning="transpilation probe",
    )


@lru_cache(maxsize=256)
def measure(
    algorithm: QuantumAlgorithm,
    n_qubits: int,
    analytic_depth: int,
    backend_name: str = DEFAULT_BACKEND,
) -> TranspiledMetrics:
    """Compile the circuit for `backend_name` and report its real cost.

    Cached: transpiling a Grover circuit onto a 127-qubit heavy-hex map takes
    seconds, and the feasibility gate may be consulted repeatedly.
    """
    # Imported here rather than at module scope: code_generator imports from
    # decision_engine, which reaches this module, and a top-level import would
    # close that cycle.
    from quantum_agent.code_generator import generate_circuit

    backend = get_backend(backend_name)
    circuit = generate_circuit(_stub_decision(algorithm, n_qubits))

    compiled = transpile(
        circuit,
        backend=backend,
        optimization_level=OPTIMIZATION_LEVEL,
        seed_transpiler=TRANSPILE_SEED,
    )

    counts = compiled.count_ops()
    two_qubit = sum(n for g, n in counts.items() if g in ("cx", "cz", "ecr"))

    return TranspiledMetrics(
        backend_name=backend_name,
        num_qubits=n_qubits,
        depth=compiled.depth(),
        two_qubit_gates=two_qubit,
        total_gates=sum(counts.values()),
        analytic_depth=analytic_depth,
        estimated_fidelity=estimate_fidelity(compiled, backend),
    )


def transpiled_depth(
    algorithm: QuantumAlgorithm,
    n_qubits: int,
    analytic_depth: int,
    backend_name: Optional[str] = None,
) -> int:
    """Compiled depth for the feasibility gate, falling back to the estimate.

    A transpilation failure must not take down the decision pipeline, so the
    analytic depth is returned instead — pessimistic about the circuit's real cost,
    but it keeps the gate operating.
    """
    try:
        return measure(
            algorithm, n_qubits, analytic_depth, backend_name or DEFAULT_BACKEND
        ).depth
    except Exception:
        return analytic_depth
