"""Agent Zero interface: natural language → quantum decision → execution.

This module is the top-level entry point that Agent Zero calls.
It accepts a free-text problem description and returns a complete
analysis + execution result.

Flow:
1. problem_analyzer parses natural language → structured features
2. decision_engine classifies advantage + gates feasibility
3. code_generator builds quantum circuit (if applicable)
4. executor runs circuit on simulator or hardware
5. Returns unified result with full reasoning chain
"""

from dataclasses import asdict
from typing import Any, Optional

from quantum_agent.problem_analyzer import analyze_problem
from quantum_agent.decision_engine import (
    ExecutionTarget,
    HardwareConstraints,
    QuantumAlgorithm,
    decide,
)
from quantum_agent.code_generator import generate_circuit
from quantum_agent.executor import execute


def quantum_decision(
    problem_description: str,
    shots: int = 1024,
    parameter_values: Optional[list[float]] = None,
    hardware: Optional[HardwareConstraints] = None,
) -> dict[str, Any]:
    """Process a natural language problem description through the full pipeline.

    This is the function that Agent Zero calls as a tool.

    Args:
        problem_description: Free-text description of the computational problem.
        shots: Number of measurement shots (default 1024).
        parameter_values: Values for parameterized circuits (QAOA, VQE).
        hardware: Hardware constraints (default: IBM Eagle 127 qubits).

    Returns:
        Dictionary with: analysis, decision, circuit_info, execution_result, summary.
    """
    # Stage 0: Analyze natural language
    analysis = analyze_problem(problem_description)

    # Stage 1-3: Decision framework
    decision = decide(
        problem_type=analysis.problem_type.value,
        search_space_size=analysis.search_space_size,
        has_oracle=analysis.has_oracle,
        has_structure=analysis.has_structure,
        num_variables=analysis.num_variables,
        hardware=hardware,
    )

    result: dict[str, Any] = {
        "analysis": {
            "problem_type": analysis.problem_type.value,
            "search_space_size": analysis.search_space_size,
            "has_oracle": analysis.has_oracle,
            "has_structure": analysis.has_structure,
            "num_variables": analysis.num_variables,
            "analyzer_confidence": analysis.confidence,
            "analyzer_reasoning": analysis.reasoning,
        },
        "decision": {
            "advantage": decision.advantage.value,
            "algorithm": decision.algorithm.value,
            "target": decision.target.value,
            "estimated_qubits": decision.estimated_qubits,
            "estimated_depth": decision.estimated_depth,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "hardware_feasible": decision.hardware_feasible,
        },
        "circuit_info": None,
        "execution_result": None,
        "summary": "",
    }

    # If classical, stop here with explanation
    if decision.algorithm == QuantumAlgorithm.NONE:
        result["summary"] = _build_classical_summary(analysis, decision)
        return result

    # Generate and execute circuit
    circuit = generate_circuit(decision)
    result["circuit_info"] = {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "num_gates": len(circuit.data),
        "num_parameters": len(circuit.parameters),
    }

    exec_result = execute(
        circuit=circuit,
        target=decision.target,
        shots=shots,
        parameter_values=parameter_values,
    )

    result["execution_result"] = {
        "counts": exec_result.counts,
        "shots": exec_result.shots,
        "target": exec_result.target.value,
        "most_likely": exec_result.most_likely,
        "success": exec_result.success,
        "error": exec_result.error,
        "metadata": exec_result.metadata,
    }

    result["summary"] = _build_quantum_summary(analysis, decision, exec_result)
    return result


def _build_classical_summary(analysis, decision) -> str:
    """Explain why classical was chosen."""
    reasons = []

    if decision.advantage.value == "classical_preferred":
        reasons.append(
            f"No known quantum advantage for {analysis.problem_type.value} problems."
        )
    elif decision.advantage.value == "no_current_advantage":
        reasons.append(
            "Quantum algorithms exist but require fault-tolerant hardware "
            "not available in the NISQ era."
        )

    if not analysis.has_oracle and analysis.problem_type.value == "unstructured_search":
        reasons.append(
            "Grover's algorithm requires a verifiable oracle, which was not "
            "identified in the problem description."
        )

    if analysis.search_space_size < 64 and analysis.problem_type.value == "unstructured_search":
        reasons.append(
            f"Search space ({analysis.search_space_size}) is too small for "
            "quantum speedup — classical brute force is faster."
        )

    return (
        f"Classical approach recommended for this {analysis.problem_type.value} problem. "
        + " ".join(reasons)
    )


def _build_quantum_summary(analysis, decision, exec_result) -> str:
    """Summarize quantum execution results."""
    if not exec_result.success:
        return (
            f"Quantum execution failed: {exec_result.error}. "
            f"The {decision.algorithm.value} algorithm was selected for this "
            f"{analysis.problem_type.value} problem."
        )

    return (
        f"Executed {decision.algorithm.value} algorithm on "
        f"{exec_result.target.value} ({exec_result.shots} shots). "
        f"Problem type: {analysis.problem_type.value}. "
        f"Advantage: {decision.advantage.value}. "
        f"Most likely result: |{exec_result.most_likely}>. "
        f"Used {decision.estimated_qubits} qubits."
    )
