"""Tests for the Quantum Decision Framework."""

import pytest
from quantum_agent.decision_engine import (
    Decision,
    ExecutionTarget,
    HardwareConstraints,
    ProblemFeatures,
    ProblemType,
    QuantumAdvantage,
    QuantumAlgorithm,
    check_feasibility,
    classify_advantage,
    decide,
    estimate_resources,
    extract_features,
    GROVER_MIN_SEARCH_SPACE,
)


# ---- Stage 1: Feature Extraction -------------------------------------------

class TestFeatureExtraction:
    def test_valid_problem_type(self):
        f = extract_features("unstructured_search", 1024, has_oracle=True)
        assert f.problem_type == ProblemType.UNSTRUCTURED_SEARCH
        assert f.search_space_size == 1024
        assert f.has_oracle is True

    def test_unknown_problem_type_defaults_to_other(self):
        f = extract_features("banana", 100)
        assert f.problem_type == ProblemType.OTHER

    def test_cryptographic_requires_fault_tolerance(self):
        f = extract_features("cryptographic", 2**20)
        assert f.requires_fault_tolerance is True

    def test_linear_algebra_requires_fault_tolerance(self):
        f = extract_features("linear_algebra", 1000)
        assert f.requires_fault_tolerance is True

    def test_search_does_not_require_fault_tolerance(self):
        f = extract_features("unstructured_search", 1024, has_oracle=True)
        assert f.requires_fault_tolerance is False


# ---- Stage 2: Quantum Advantage Classification ----------------------------

class TestClassification:
    def test_grover_clear_advantage(self):
        f = ProblemFeatures(
            problem_type=ProblemType.UNSTRUCTURED_SEARCH,
            search_space_size=1024,
            has_oracle=True,
        )
        advantage, algo, conf = classify_advantage(f)
        assert advantage == QuantumAdvantage.CLEAR
        assert algo == QuantumAlgorithm.GROVER

    def test_grover_no_oracle_falls_to_classical(self):
        f = ProblemFeatures(
            problem_type=ProblemType.UNSTRUCTURED_SEARCH,
            search_space_size=1024,
            has_oracle=False,
        )
        advantage, algo, _ = classify_advantage(f)
        assert advantage == QuantumAdvantage.CLASSICAL_PREFERRED
        assert algo == QuantumAlgorithm.NONE

    def test_grover_small_search_space_classical(self):
        f = ProblemFeatures(
            problem_type=ProblemType.UNSTRUCTURED_SEARCH,
            search_space_size=GROVER_MIN_SEARCH_SPACE - 1,
            has_oracle=True,
        )
        advantage, algo, _ = classify_advantage(f)
        assert advantage == QuantumAdvantage.CLASSICAL_PREFERRED

    def test_qaoa_potential_advantage(self):
        f = ProblemFeatures(
            problem_type=ProblemType.COMBINATORIAL_OPTIMIZATION,
            search_space_size=2**10,
            num_variables=10,
        )
        advantage, algo, _ = classify_advantage(f)
        assert advantage == QuantumAdvantage.POTENTIAL
        assert algo == QuantumAlgorithm.QAOA

    def test_vqe_clear_advantage(self):
        f = ProblemFeatures(
            problem_type=ProblemType.QUANTUM_SIMULATION,
            search_space_size=256,
            num_variables=4,
        )
        advantage, algo, _ = classify_advantage(f)
        assert advantage == QuantumAdvantage.CLEAR
        assert algo == QuantumAlgorithm.VQE

    def test_cryptographic_no_current_advantage(self):
        f = ProblemFeatures(
            problem_type=ProblemType.CRYPTOGRAPHIC,
            search_space_size=2**20,
            requires_fault_tolerance=True,
        )
        advantage, algo, _ = classify_advantage(f)
        assert advantage == QuantumAdvantage.NO_CURRENT
        assert algo == QuantumAlgorithm.NONE

    def test_ml_classical_preferred(self):
        f = ProblemFeatures(
            problem_type=ProblemType.MACHINE_LEARNING,
            search_space_size=10000,
        )
        advantage, algo, _ = classify_advantage(f)
        assert advantage == QuantumAdvantage.CLASSICAL_PREFERRED

    def test_unknown_type_classical(self):
        f = ProblemFeatures(
            problem_type=ProblemType.OTHER,
            search_space_size=100,
        )
        advantage, _, _ = classify_advantage(f)
        assert advantage == QuantumAdvantage.CLASSICAL_PREFERRED


# ---- Stage 3: Hardware Feasibility Gate ------------------------------------

class TestResourceEstimation:
    def test_grover_qubits(self):
        f = ProblemFeatures(
            problem_type=ProblemType.UNSTRUCTURED_SEARCH,
            search_space_size=1024,
            has_oracle=True,
        )
        qubits, depth = estimate_resources(QuantumAlgorithm.GROVER, f)
        assert qubits == 10  # log2(1024) = 10
        assert depth > 0

    def test_qaoa_uses_num_variables(self):
        f = ProblemFeatures(
            problem_type=ProblemType.COMBINATORIAL_OPTIMIZATION,
            search_space_size=2**10,
            num_variables=10,
        )
        qubits, depth = estimate_resources(QuantumAlgorithm.QAOA, f)
        assert qubits == 10
        assert depth == 20  # 2 * n_qubits

    def test_vqe_default_qubits(self):
        f = ProblemFeatures(
            problem_type=ProblemType.QUANTUM_SIMULATION,
            search_space_size=256,
        )
        qubits, depth = estimate_resources(QuantumAlgorithm.VQE, f)
        assert qubits == 4  # default
        assert depth == 16  # 4 * 4

    def test_none_algorithm_returns_zero(self):
        f = ProblemFeatures(
            problem_type=ProblemType.OTHER,
            search_space_size=100,
        )
        qubits, depth = estimate_resources(QuantumAlgorithm.NONE, f)
        assert qubits == 0
        assert depth == 0


class TestFeasibility:
    def test_small_circuit_fits_hardware(self):
        feasible, target = check_feasibility(10, 50)
        assert feasible is True
        assert target == ExecutionTarget.QUANTUM_HARDWARE

    def test_too_many_qubits_falls_to_simulator(self):
        hw = HardwareConstraints(max_qubits=20, max_circuit_depth=100)
        feasible, target = check_feasibility(25, 50, hw)
        assert feasible is False
        assert target == ExecutionTarget.QUANTUM_SIMULATE

    def test_too_deep_falls_to_simulator(self):
        hw = HardwareConstraints(max_qubits=127, max_circuit_depth=50)
        feasible, target = check_feasibility(10, 200, hw)
        assert feasible is False
        assert target == ExecutionTarget.QUANTUM_SIMULATE

    def test_huge_circuit_falls_to_classical(self):
        feasible, target = check_feasibility(50, 10000)
        assert feasible is False
        assert target == ExecutionTarget.CLASSICAL

    def test_zero_qubits_is_classical(self):
        feasible, target = check_feasibility(0, 0)
        assert feasible is False
        assert target == ExecutionTarget.CLASSICAL


# ---- Full Pipeline (decide) ------------------------------------------------

class TestDecide:
    def test_grover_end_to_end(self):
        d = decide("unstructured_search", 1024, has_oracle=True)
        assert d.advantage == QuantumAdvantage.CLEAR
        assert d.algorithm == QuantumAlgorithm.GROVER
        assert d.estimated_qubits == 10
        assert d.target in (ExecutionTarget.QUANTUM_HARDWARE, ExecutionTarget.QUANTUM_SIMULATE)

    def test_classical_problem_short_circuits(self):
        d = decide("machine_learning", 10000)
        assert d.algorithm == QuantumAlgorithm.NONE
        assert d.target == ExecutionTarget.CLASSICAL
        assert d.estimated_qubits == 0

    def test_qaoa_with_custom_hardware(self):
        hw = HardwareConstraints(max_qubits=5, max_circuit_depth=10)
        d = decide("combinatorial_optimization", 2**10, num_variables=10, hardware=hw)
        assert d.algorithm == QuantumAlgorithm.QAOA
        assert d.hardware_feasible is False
        assert d.target == ExecutionTarget.QUANTUM_SIMULATE

    def test_decision_has_reasoning(self):
        d = decide("unstructured_search", 1024, has_oracle=True)
        assert len(d.reasoning) > 0
        assert "unstructured_search" in d.reasoning

    def test_cryptographic_rejected(self):
        d = decide("cryptographic", 2**20)
        assert d.advantage == QuantumAdvantage.NO_CURRENT
        assert d.target == ExecutionTarget.CLASSICAL
