"""Generates Qiskit quantum circuits based on decision engine output.

Supported algorithms:
- Grover's search (unstructured search with oracle)
- QAOA (combinatorial optimization — MaxCut)
- VQE (quantum simulation with parameterized ansatz)
"""

from math import ceil, pi, sqrt

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from quantum_agent.decision_engine import Decision, QuantumAlgorithm


class UnsupportedAlgorithmError(Exception):
    pass


def generate_circuit(decision: Decision) -> QuantumCircuit:
    """Generate a Qiskit circuit based on the decision engine output.

    This is the main entry point for code generation.
    """
    generators = {
        QuantumAlgorithm.GROVER: _build_grover_circuit,
        QuantumAlgorithm.QAOA: _build_qaoa_circuit,
        QuantumAlgorithm.VQE: _build_vqe_circuit,
    }

    if decision.algorithm not in generators:
        raise UnsupportedAlgorithmError(
            f"No circuit generator for algorithm: {decision.algorithm.value}"
        )

    return generators[decision.algorithm](decision)


# ---------------------------------------------------------------------------
# Grover's Algorithm
# ---------------------------------------------------------------------------

def _build_grover_oracle(n_qubits: int, marked_state: int) -> QuantumCircuit:
    """Build an oracle that flips the phase of |marked_state>.

    Uses a multi-controlled Z gate pattern:
    - Apply X to qubits where the marked state has a 0 bit
    - Apply multi-controlled Z (MCZ via H-MCX-H on target)
    - Undo the X gates
    """
    oracle = QuantumCircuit(n_qubits, name="oracle")

    # Flip qubits where marked_state has bit = 0
    for i in range(n_qubits):
        if not (marked_state >> i) & 1:
            oracle.x(i)

    # Multi-controlled Z: H on last qubit, MCX, H on last qubit
    oracle.h(n_qubits - 1)
    oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    oracle.h(n_qubits - 1)

    # Undo X flips
    for i in range(n_qubits):
        if not (marked_state >> i) & 1:
            oracle.x(i)

    return oracle


def _build_grover_diffuser(n_qubits: int) -> QuantumCircuit:
    """Build the Grover diffusion operator (amplitude amplification)."""
    diffuser = QuantumCircuit(n_qubits, name="diffuser")

    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))

    # Multi-controlled Z
    diffuser.h(n_qubits - 1)
    diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffuser.h(n_qubits - 1)

    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))

    return diffuser


def _build_grover_circuit(decision: Decision, marked_state: int = 0) -> QuantumCircuit:
    """Build a complete Grover's search circuit.

    Args:
        decision: Output from the decision engine.
        marked_state: The state to search for (default 0, i.e. |00...0>
                      is the target after oracle phase flip).
    """
    n_qubits = decision.estimated_qubits
    search_space = 2 ** n_qubits
    n_iterations = max(1, round(pi / 4 * sqrt(search_space)))

    qc = QuantumCircuit(n_qubits, n_qubits, name="grover")

    # Initialize uniform superposition
    qc.h(range(n_qubits))
    qc.barrier()

    # Grover iterations
    oracle = _build_grover_oracle(n_qubits, marked_state)
    diffuser = _build_grover_diffuser(n_qubits)

    for _ in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
        qc.barrier()

    # Measure
    qc.measure(range(n_qubits), range(n_qubits))

    return qc


# ---------------------------------------------------------------------------
# QAOA (MaxCut)
# ---------------------------------------------------------------------------

def _build_qaoa_circuit(
    decision: Decision,
    edges: list[tuple[int, int]] | None = None,
    p: int = 1,
) -> QuantumCircuit:
    """Build a QAOA circuit for MaxCut.

    Args:
        decision: Output from the decision engine.
        edges: Graph edges as list of (i, j) tuples. If None, uses a default
               ring graph (0-1, 1-2, ..., (n-1)-0).
        p: Number of QAOA layers (default 1).
    """
    n_qubits = decision.estimated_qubits

    # Default: ring graph
    if edges is None:
        edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

    qc = QuantumCircuit(n_qubits, n_qubits, name="qaoa_maxcut")

    # Initial superposition
    qc.h(range(n_qubits))

    for layer in range(p):
        gamma = Parameter(f"γ_{layer}")
        beta = Parameter(f"β_{layer}")

        # Cost layer: ZZ interaction for each edge
        for i, j in edges:
            qc.cx(i, j)
            qc.rz(2 * gamma, j)
            qc.cx(i, j)

        qc.barrier()

        # Mixer layer: X rotations
        for i in range(n_qubits):
            qc.rx(2 * beta, i)

        qc.barrier()

    # Measure
    qc.measure(range(n_qubits), range(n_qubits))

    return qc


# ---------------------------------------------------------------------------
# VQE (Variational Quantum Eigensolver)
# ---------------------------------------------------------------------------

def _build_vqe_circuit(decision: Decision, depth: int = 1) -> QuantumCircuit:
    """Build a hardware-efficient VQE ansatz.

    Uses Ry-CNOT ladder structure — a standard choice for NISQ devices.

    Args:
        decision: Output from the decision engine.
        depth: Number of entangling layers (default 1).
    """
    n_qubits = decision.estimated_qubits

    qc = QuantumCircuit(n_qubits, n_qubits, name="vqe_ansatz")

    param_idx = 0
    for d in range(depth):
        # Rotation layer
        for i in range(n_qubits):
            theta = Parameter(f"θ_{param_idx}")
            qc.ry(theta, i)
            param_idx += 1

        # Entangling layer (linear CNOT ladder)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

        qc.barrier()

    # Final rotation layer
    for i in range(n_qubits):
        theta = Parameter(f"θ_{param_idx}")
        qc.ry(theta, i)
        param_idx += 1

    # Measure
    qc.measure(range(n_qubits), range(n_qubits))

    return qc
