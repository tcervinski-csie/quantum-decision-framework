"""Agent Zero tool wrapper: ties decision → generation → execution.

This module provides the interface that Agent Zero calls as a custom tool.
It takes a natural-language-style problem description (structured as kwargs),
runs it through the full pipeline, and returns a result dictionary.
"""

from dataclasses import asdict
from typing import Any, Optional

from quantum_agent.decision_engine import (
    Decision,
    ExecutionTarget,
    HardwareConstraints,
    QuantumAlgorithm,
    decide,
)
from quantum_agent.code_generator import generate_circuit
from quantum_agent.executor import execute, ExecutionResult


def quantum_tool(
    problem_type: str,
    search_space_size: int,
    has_oracle: bool = False,
    has_structure: bool = False,
    num_variables: int = 0,
    shots: int = 1024,
    parameter_values: Optional[list[float]] = None,
    hardware: Optional[dict] = None,
) -> dict[str, Any]:
    """Agent Zero tool entry point.

    This is the function that Agent Zero calls when it wants to evaluate
    and optionally execute a quantum computation.

    Args:
        problem_type: One of "unstructured_search", "combinatorial_optimization",
                      "quantum_simulation", "cryptographic", "linear_algebra",
                      "machine_learning", or "other".
        search_space_size: Size of the solution space.
        has_oracle: Whether a verification oracle exists.
        has_structure: Whether the problem has exploitable mathematical structure.
        num_variables: Number of problem variables (used for qubit estimation).
        shots: Number of measurement shots for execution.
        parameter_values: Parameter values for variational circuits (QAOA, VQE).
        hardware: Optional hardware constraints as dict with keys:
                  max_qubits, max_circuit_depth, connectivity.

    Returns:
        Dictionary with keys: decision, circuit_info, execution_result, summary.
    """
    # Parse hardware constraints
    hw = None
    if hardware:
        hw = HardwareConstraints(
            max_qubits=hardware.get("max_qubits", 127),
            max_circuit_depth=hardware.get("max_circuit_depth", 100),
            connectivity=hardware.get("connectivity", "heavy_hex"),
        )

    # Stage 1-3: Decision
    decision = decide(
        problem_type=problem_type,
        search_space_size=search_space_size,
        has_oracle=has_oracle,
        has_structure=has_structure,
        num_variables=num_variables,
        hardware=hw,
    )

    result: dict[str, Any] = {
        "decision": asdict(decision),
        "circuit_info": None,
        "execution_result": None,
        "summary": "",
    }

    # Serialize enums in decision
    result["decision"]["advantage"] = decision.advantage.value
    result["decision"]["algorithm"] = decision.algorithm.value
    result["decision"]["target"] = decision.target.value

    # If classical, stop here
    if decision.algorithm == QuantumAlgorithm.NONE:
        result["summary"] = (
            f"Classical approach recommended. {decision.reasoning}"
        )
        return result

    # Generate circuit
    circuit = generate_circuit(decision)
    result["circuit_info"] = {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "num_gates": len(circuit.data),
        "num_parameters": len(circuit.parameters),
    }

    # Execute
    exec_result = execute(
        circuit=circuit,
        target=decision.target,
        shots=shots,
        parameter_values=parameter_values,
    )

    result["execution_result"] = asdict(exec_result)
    result["execution_result"]["target"] = exec_result.target.value

    # Build summary
    if exec_result.success:
        result["summary"] = (
            f"Executed {decision.algorithm.value} on {exec_result.target.value} "
            f"with {exec_result.shots} shots. "
            f"Most likely result: {exec_result.most_likely}. "
            f"Advantage: {decision.advantage.value}."
        )
    else:
        result["summary"] = (
            f"Execution failed: {exec_result.error}. "
            f"Decision was {decision.algorithm.value} on {decision.target.value}."
        )

    return result
