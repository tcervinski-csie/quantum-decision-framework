"""Tests for the quantum circuit executor."""

import pytest
from math import pi
from qiskit import QuantumCircuit

from quantum_agent.executor import (
    ExecutionResult,
    bind_parameters,
    execute,
    execute_on_simulator,
)
from quantum_agent.decision_engine import (
    Decision,
    ExecutionTarget,
    QuantumAdvantage,
    QuantumAlgorithm,
)
from quantum_agent.code_generator import generate_circuit


def _make_decision(algorithm, qubits, target=ExecutionTarget.QUANTUM_SIMULATE):
    return Decision(
        advantage=QuantumAdvantage.CLEAR,
        algorithm=algorithm,
        target=target,
        estimated_qubits=qubits,
        estimated_depth=20,
        confidence=0.9,
        reasoning="test",
        hardware_feasible=True,
    )


# ---- Parameter Binding ------------------------------------------------------

class TestParameterBinding:
    def test_bind_correct_count(self):
        from qiskit.circuit import Parameter
        qc = QuantumCircuit(2)
        p0, p1 = Parameter("a"), Parameter("b")
        qc.ry(p0, 0)
        qc.ry(p1, 1)
        qc.measure_all()

        bound = bind_parameters(qc, [pi / 2, pi / 4])
        assert len(bound.parameters) == 0

    def test_bind_wrong_count_raises(self):
        from qiskit.circuit import Parameter
        qc = QuantumCircuit(1)
        qc.ry(Parameter("a"), 0)
        qc.measure_all()

        with pytest.raises(ValueError, match="Expected 1"):
            bind_parameters(qc, [0.1, 0.2])


# ---- Simulator Execution ----------------------------------------------------

class TestSimulatorExecution:
    def test_simple_bell_state(self):
        """Bell state: should produce ~50% |00> and ~50% |11>."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        result = execute_on_simulator(qc, shots=1000)
        assert result.success is True
        assert result.target == ExecutionTarget.QUANTUM_SIMULATE
        assert result.shots == 1000
        # Bell state should only produce "00" and "11"
        for key in result.counts:
            assert key in ("00", "11")

    def test_grover_finds_marked_state(self):
        """Grover's on 3 qubits searching for |000> should find it."""
        d = _make_decision(QuantumAlgorithm.GROVER, 3)
        qc = generate_circuit(d)

        result = execute_on_simulator(qc, shots=1024)
        assert result.success is True
        assert len(result.counts) > 0
        # The marked state (0) should be the most likely
        assert result.most_likely == "000"

    def test_qaoa_runs_with_parameters(self):
        """QAOA circuit should execute after binding parameters."""
        d = _make_decision(QuantumAlgorithm.QAOA, 4)
        qc = generate_circuit(d)

        # p=1: 1 gamma + 1 beta
        result = execute_on_simulator(qc, parameter_values=[0.5, 0.5])
        assert result.success is True
        assert result.shots == 1024
        assert len(result.counts) > 0

    def test_vqe_runs_with_parameters(self):
        """VQE ansatz should execute after binding parameters."""
        d = _make_decision(QuantumAlgorithm.VQE, 4)
        qc = generate_circuit(d)

        # 8 parameters for 4-qubit depth-1 ansatz
        values = [0.1] * 8
        result = execute_on_simulator(qc, parameter_values=values)
        assert result.success is True
        assert len(result.counts) > 0

    def test_unbound_parameters_fail_gracefully(self):
        """Circuit with parameters but no values should return error."""
        d = _make_decision(QuantumAlgorithm.QAOA, 4)
        qc = generate_circuit(d)

        result = execute_on_simulator(qc)
        assert result.success is False
        assert result.error is not None

    def test_result_metadata(self):
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)

        result = execute_on_simulator(qc)
        assert result.metadata["simulator"] == "aer"


# ---- Execute Router ---------------------------------------------------------

class TestExecuteRouter:
    def test_routes_to_simulator(self):
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)

        result = execute(qc, ExecutionTarget.QUANTUM_SIMULATE)
        assert result.success is True
        assert result.most_likely == "1"

    def test_classical_target_returns_error(self):
        qc = QuantumCircuit(1, 1)
        result = execute(qc, ExecutionTarget.CLASSICAL)
        assert result.success is False
        assert "Classical" in result.error

    def test_hardware_without_credentials_fails_gracefully(self):
        """Hardware execution without IBM credentials should fail, not crash."""
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)

        result = execute(qc, ExecutionTarget.QUANTUM_HARDWARE)
        assert result.success is False
        assert result.error is not None
