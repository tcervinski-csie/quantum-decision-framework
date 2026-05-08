"""End-to-end integration tests: problem in → result out via agent_tool."""

import pytest

from quantum_agent.agent_tool import quantum_tool


class TestEndToEnd:
    def test_grover_full_pipeline(self):
        """Unstructured search → Grover's → simulate → result."""
        result = quantum_tool(
            problem_type="unstructured_search",
            search_space_size=1024,
            has_oracle=True,
        )
        assert result["decision"]["algorithm"] == "grover"
        assert result["decision"]["advantage"] == "clear_advantage"
        assert result["circuit_info"] is not None
        assert result["circuit_info"]["num_qubits"] == 10
        assert result["execution_result"]["success"] is True
        assert result["execution_result"]["most_likely"] != ""
        assert "grover" in result["summary"]

    def test_qaoa_full_pipeline(self):
        """Combinatorial optimization → QAOA → simulate → result."""
        result = quantum_tool(
            problem_type="combinatorial_optimization",
            search_space_size=2**6,
            num_variables=6,
            parameter_values=[0.5, 0.5],
            hardware={"max_qubits": 0, "max_circuit_depth": 0},
        )
        assert result["decision"]["algorithm"] == "qaoa"
        assert result["decision"]["advantage"] == "potential_advantage"
        assert result["circuit_info"]["num_qubits"] == 6
        assert result["execution_result"]["success"] is True

    def test_vqe_full_pipeline(self):
        """Quantum simulation → VQE → simulate → result."""
        result = quantum_tool(
            problem_type="quantum_simulation",
            search_space_size=16,
            num_variables=4,
            parameter_values=[0.1] * 8,
            hardware={"max_qubits": 0, "max_circuit_depth": 0},
        )
        assert result["decision"]["algorithm"] == "vqe"
        assert result["decision"]["advantage"] == "clear_advantage"
        assert result["execution_result"]["success"] is True

    def test_classical_problem_no_execution(self):
        """ML problem → classical → no circuit, no execution."""
        result = quantum_tool(
            problem_type="machine_learning",
            search_space_size=10000,
        )
        assert result["decision"]["algorithm"] == "none"
        assert result["circuit_info"] is None
        assert result["execution_result"] is None
        assert "Classical" in result["summary"]

    def test_cryptographic_rejected(self):
        """Cryptographic problem → rejected (needs fault tolerance)."""
        result = quantum_tool(
            problem_type="cryptographic",
            search_space_size=2**20,
        )
        assert result["decision"]["advantage"] == "no_current_advantage"
        assert result["circuit_info"] is None

    def test_small_search_space_stays_classical(self):
        """Tiny search space → classical even with oracle."""
        result = quantum_tool(
            problem_type="unstructured_search",
            search_space_size=8,
            has_oracle=True,
        )
        assert result["decision"]["algorithm"] == "none"
        assert result["decision"]["advantage"] == "classical_preferred"

    def test_hardware_constraints_force_simulator(self):
        """Problem fits quantum but exceeds hardware → falls to simulator."""
        result = quantum_tool(
            problem_type="combinatorial_optimization",
            search_space_size=2**10,
            num_variables=10,
            parameter_values=[0.5, 0.5],
            hardware={"max_qubits": 5, "max_circuit_depth": 10},
        )
        assert result["decision"]["target"] == "quantum_simulate"
        assert result["execution_result"]["success"] is True

    def test_unknown_problem_type(self):
        """Unknown problem type → classical."""
        result = quantum_tool(
            problem_type="time_travel",
            search_space_size=999,
        )
        assert result["decision"]["algorithm"] == "none"
        assert result["decision"]["advantage"] == "classical_preferred"

    def test_qaoa_without_parameters_fails_gracefully(self):
        """QAOA without parameter values should fail but not crash."""
        result = quantum_tool(
            problem_type="combinatorial_optimization",
            search_space_size=2**4,
            num_variables=4,
        )
        assert result["execution_result"]["success"] is False
        assert result["execution_result"]["error"] is not None
