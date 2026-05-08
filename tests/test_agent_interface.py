"""Tests for the Agent Zero interface — natural language end-to-end."""

import pytest
from quantum_agent.agent_interface import quantum_decision
from quantum_agent.decision_engine import HardwareConstraints


class TestNaturalLanguagePipeline:
    """Test that natural language descriptions flow through the full pipeline."""

    def test_grover_from_text(self):
        result = quantum_decision(
            "Search through an unsorted database of 1024 elements "
            "to find one that satisfies a verification condition."
        )
        assert result["analysis"]["problem_type"] == "unstructured_search"
        assert result["analysis"]["has_oracle"] is True
        assert result["decision"]["algorithm"] == "grover"
        assert result["decision"]["advantage"] == "clear_advantage"
        assert result["execution_result"]["success"] is True
        assert result["execution_result"]["most_likely"] != ""

    def test_maxcut_from_text(self):
        result = quantum_decision(
            "Find the optimal MaxCut partition for a graph with 6 nodes.",
            parameter_values=[0.5, 0.5],
            hardware=HardwareConstraints(max_qubits=0, max_circuit_depth=0),
        )
        assert result["analysis"]["problem_type"] == "combinatorial_optimization"
        assert result["decision"]["algorithm"] == "qaoa"
        assert result["circuit_info"]["num_qubits"] == 6
        assert result["execution_result"]["success"] is True

    def test_molecule_simulation_from_text(self):
        result = quantum_decision(
            "Simulate the ground state energy of a molecule using 4 qubits.",
            parameter_values=[0.1] * 8,
            hardware=HardwareConstraints(max_qubits=0, max_circuit_depth=0),
        )
        assert result["analysis"]["problem_type"] == "quantum_simulation"
        assert result["decision"]["algorithm"] == "vqe"
        assert result["execution_result"]["success"] is True

    def test_cryptographic_rejected_from_text(self):
        result = quantum_decision(
            "Factor a large RSA number into its prime factors."
        )
        assert result["analysis"]["problem_type"] == "cryptographic"
        assert result["decision"]["advantage"] == "no_current_advantage"
        assert result["decision"]["algorithm"] == "none"
        assert result["circuit_info"] is None
        assert "fault-tolerant" in result["summary"]

    def test_ml_stays_classical_from_text(self):
        result = quantum_decision(
            "Train a deep learning classification model on image data."
        )
        assert result["analysis"]["problem_type"] == "machine_learning"
        assert result["decision"]["algorithm"] == "none"
        assert "Classical" in result["summary"]

    def test_small_search_stays_classical(self):
        result = quantum_decision(
            "Search through 10 elements to find the one that satisfies a constraint."
        )
        assert result["decision"]["algorithm"] == "none"
        assert result["decision"]["advantage"] == "classical_preferred"

    def test_unknown_problem_stays_classical(self):
        result = quantum_decision("What is the meaning of life?")
        assert result["decision"]["algorithm"] == "none"
        assert result["circuit_info"] is None


class TestReasoningChain:
    """Verify the full reasoning chain is present in results."""

    def test_analysis_has_reasoning(self):
        result = quantum_decision(
            "Search a database of 5000 items with a verification oracle."
        )
        assert len(result["analysis"]["analyzer_reasoning"]) > 0
        assert result["analysis"]["analyzer_confidence"] > 0

    def test_decision_has_reasoning(self):
        result = quantum_decision(
            "Optimize a portfolio of 8 assets to maximize returns."
        )
        assert len(result["decision"]["reasoning"]) > 0

    def test_summary_present(self):
        result = quantum_decision(
            "Search 2048 records to find one matching a condition."
        )
        assert len(result["summary"]) > 0


class TestParameterHandling:
    """Test variational circuit parameter handling through the agent."""

    def test_qaoa_missing_parameters_fails_gracefully(self):
        result = quantum_decision(
            "Optimize the MaxCut of a 4 node graph.",
            hardware=HardwareConstraints(max_qubits=0, max_circuit_depth=0),
        )
        assert result["decision"]["algorithm"] == "qaoa"
        assert result["execution_result"]["success"] is False
        assert result["execution_result"]["error"] is not None

    def test_vqe_with_parameters_succeeds(self):
        result = quantum_decision(
            "Find the ground state energy of a 4 qubit Hamiltonian.",
            parameter_values=[0.2] * 8,
            hardware=HardwareConstraints(max_qubits=0, max_circuit_depth=0),
        )
        assert result["execution_result"]["success"] is True
