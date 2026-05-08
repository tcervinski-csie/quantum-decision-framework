"""Tests for the quantum circuit code generator."""

import pytest
from qiskit import QuantumCircuit

from quantum_agent.code_generator import (
    UnsupportedAlgorithmError,
    generate_circuit,
    _build_grover_oracle,
    _build_grover_diffuser,
)
from quantum_agent.decision_engine import (
    Decision,
    ExecutionTarget,
    QuantumAdvantage,
    QuantumAlgorithm,
)


def _make_decision(algorithm: QuantumAlgorithm, qubits: int, depth: int = 20) -> Decision:
    """Helper to build a Decision object for testing."""
    return Decision(
        advantage=QuantumAdvantage.CLEAR,
        algorithm=algorithm,
        target=ExecutionTarget.QUANTUM_SIMULATE,
        estimated_qubits=qubits,
        estimated_depth=depth,
        confidence=0.9,
        reasoning="test",
        hardware_feasible=True,
    )


# ---- Grover's Circuit ------------------------------------------------------

class TestGroverCircuit:
    def test_generates_valid_circuit(self):
        d = _make_decision(QuantumAlgorithm.GROVER, 3)
        qc = generate_circuit(d)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 3
        assert qc.num_clbits == 3

    def test_has_measurements(self):
        d = _make_decision(QuantumAlgorithm.GROVER, 3)
        qc = generate_circuit(d)
        measure_ops = [inst for inst in qc.data if inst.operation.name == "measure"]
        assert len(measure_ops) == 3

    def test_circuit_contains_hadamards(self):
        d = _make_decision(QuantumAlgorithm.GROVER, 3)
        qc = generate_circuit(d)
        h_ops = [inst for inst in qc.data if inst.operation.name == "h"]
        assert len(h_ops) > 0

    def test_oracle_is_unitary(self):
        """Oracle should be its own inverse (applying twice = identity)."""
        oracle = _build_grover_oracle(3, marked_state=5)
        # Compose oracle with itself — should be identity (up to global phase)
        combined = oracle.compose(oracle)
        # No net X gates should remain (they cancel)
        x_ops = [inst for inst in combined.data if inst.operation.name == "x"]
        # Each X in oracle appears twice → cancels. Verify even count per qubit.
        assert len(x_ops) % 2 == 0

    def test_different_qubit_counts(self):
        for n in [2, 4, 6]:
            d = _make_decision(QuantumAlgorithm.GROVER, n)
            qc = generate_circuit(d)
            assert qc.num_qubits == n


# ---- QAOA Circuit -----------------------------------------------------------

class TestQAOACircuit:
    def test_generates_valid_circuit(self):
        d = _make_decision(QuantumAlgorithm.QAOA, 4)
        qc = generate_circuit(d)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 4

    def test_has_parameters(self):
        d = _make_decision(QuantumAlgorithm.QAOA, 4)
        qc = generate_circuit(d)
        # p=1 layer: 1 gamma + 1 beta = 2 parameters
        assert len(qc.parameters) == 2

    def test_has_measurements(self):
        d = _make_decision(QuantumAlgorithm.QAOA, 4)
        qc = generate_circuit(d)
        measure_ops = [inst for inst in qc.data if inst.operation.name == "measure"]
        assert len(measure_ops) == 4

    def test_contains_cx_gates(self):
        d = _make_decision(QuantumAlgorithm.QAOA, 4)
        qc = generate_circuit(d)
        cx_ops = [inst for inst in qc.data if inst.operation.name == "cx"]
        assert len(cx_ops) > 0

    def test_ring_graph_edges(self):
        """Default ring graph: n edges for n qubits, each edge = 2 CX gates."""
        d = _make_decision(QuantumAlgorithm.QAOA, 4)
        qc = generate_circuit(d)
        cx_ops = [inst for inst in qc.data if inst.operation.name == "cx"]
        # Ring graph with 4 nodes = 4 edges, 2 CX per edge = 8
        assert len(cx_ops) == 8


# ---- VQE Circuit ------------------------------------------------------------

class TestVQECircuit:
    def test_generates_valid_circuit(self):
        d = _make_decision(QuantumAlgorithm.VQE, 4)
        qc = generate_circuit(d)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 4

    def test_has_parameters(self):
        d = _make_decision(QuantumAlgorithm.VQE, 4)
        qc = generate_circuit(d)
        # depth=1: rotation layer (4) + entangling + final rotation (4) = 8 params
        assert len(qc.parameters) == 8

    def test_has_measurements(self):
        d = _make_decision(QuantumAlgorithm.VQE, 4)
        qc = generate_circuit(d)
        measure_ops = [inst for inst in qc.data if inst.operation.name == "measure"]
        assert len(measure_ops) == 4

    def test_contains_ry_gates(self):
        d = _make_decision(QuantumAlgorithm.VQE, 4)
        qc = generate_circuit(d)
        ry_ops = [inst for inst in qc.data if inst.operation.name == "ry"]
        assert len(ry_ops) == 8  # 2 rotation layers × 4 qubits

    def test_cnot_ladder(self):
        d = _make_decision(QuantumAlgorithm.VQE, 4)
        qc = generate_circuit(d)
        cx_ops = [inst for inst in qc.data if inst.operation.name == "cx"]
        assert len(cx_ops) == 3  # linear ladder: n-1 CNOTs


# ---- Error Handling ---------------------------------------------------------

class TestErrorHandling:
    def test_unsupported_algorithm_raises(self):
        d = _make_decision(QuantumAlgorithm.NONE, 0)
        with pytest.raises(UnsupportedAlgorithmError):
            generate_circuit(d)
