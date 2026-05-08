"""Execute quantum circuits on Aer simulator or IBM Quantum hardware.

Handles:
- Binding parameterized circuits (QAOA, VQE) with provided values
- Running on Aer simulator locally
- Running on IBM Quantum hardware via qiskit-ibm-runtime
- Returning structured results with counts and metadata
"""

from dataclasses import dataclass, field
from typing import Optional

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from quantum_agent.decision_engine import ExecutionTarget


@dataclass
class ExecutionResult:
    """Result from executing a quantum circuit."""
    counts: dict[str, int]
    shots: int
    target: ExecutionTarget
    most_likely: str
    success: bool
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def bind_parameters(circuit: QuantumCircuit, values: list[float]) -> QuantumCircuit:
    """Bind parameter values to a parameterized circuit.

    Args:
        circuit: A parameterized QuantumCircuit (e.g., QAOA or VQE).
        values: List of float values, one per parameter in circuit order.

    Returns:
        A new circuit with all parameters bound.
    """
    params = list(circuit.parameters)
    if len(values) != len(params):
        raise ValueError(
            f"Expected {len(params)} parameter values, got {len(values)}"
        )
    param_dict = dict(zip(params, values))
    return circuit.assign_parameters(param_dict)


def execute_on_simulator(
    circuit: QuantumCircuit,
    shots: int = 1024,
    parameter_values: Optional[list[float]] = None,
) -> ExecutionResult:
    """Run a circuit on the local Aer simulator.

    Args:
        circuit: The quantum circuit to execute.
        shots: Number of measurement shots (default 1024).
        parameter_values: If the circuit has parameters, provide values here.
    """
    try:
        # Bind parameters if needed
        if circuit.parameters:
            if parameter_values is None:
                raise ValueError(
                    "Circuit has unbound parameters. Provide parameter_values."
                )
            circuit = bind_parameters(circuit, parameter_values)

        simulator = AerSimulator()
        result = simulator.run(circuit, shots=shots).result()
        counts = result.get_counts(circuit)

        # Find most likely outcome
        most_likely = max(counts, key=counts.get)

        return ExecutionResult(
            counts=counts,
            shots=shots,
            target=ExecutionTarget.QUANTUM_SIMULATE,
            most_likely=most_likely,
            success=True,
            metadata={"simulator": "aer", "method": "automatic"},
        )
    except Exception as e:
        return ExecutionResult(
            counts={},
            shots=shots,
            target=ExecutionTarget.QUANTUM_SIMULATE,
            most_likely="",
            success=False,
            error=str(e),
        )


def execute_on_hardware(
    circuit: QuantumCircuit,
    shots: int = 1024,
    parameter_values: Optional[list[float]] = None,
    backend_name: str = "ibm_brisbane",
) -> ExecutionResult:
    """Run a circuit on IBM Quantum hardware via qiskit-ibm-runtime.

    Requires IBM Quantum credentials to be configured
    (e.g., via QISKIT_IBM_TOKEN environment variable).

    Args:
        circuit: The quantum circuit to execute.
        shots: Number of measurement shots.
        parameter_values: If the circuit has parameters, provide values here.
        backend_name: IBM backend name (default: ibm_brisbane).
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        # Bind parameters if needed
        if circuit.parameters:
            if parameter_values is None:
                raise ValueError(
                    "Circuit has unbound parameters. Provide parameter_values."
                )
            circuit = bind_parameters(circuit, parameter_values)

        service = QiskitRuntimeService()
        backend = service.backend(backend_name)

        sampler = SamplerV2(backend)
        job = sampler.run([circuit], shots=shots)
        result = job.result()

        # Extract counts from SamplerV2 result
        pub_result = result[0]
        counts = pub_result.data.meas.get_counts()

        most_likely = max(counts, key=counts.get)

        return ExecutionResult(
            counts=counts,
            shots=shots,
            target=ExecutionTarget.QUANTUM_HARDWARE,
            most_likely=most_likely,
            success=True,
            metadata={
                "backend": backend_name,
                "job_id": job.job_id(),
            },
        )
    except Exception as e:
        return ExecutionResult(
            counts={},
            shots=shots,
            target=ExecutionTarget.QUANTUM_HARDWARE,
            most_likely="",
            success=False,
            error=str(e),
        )


def execute(
    circuit: QuantumCircuit,
    target: ExecutionTarget,
    shots: int = 1024,
    parameter_values: Optional[list[float]] = None,
    backend_name: str = "ibm_brisbane",
) -> ExecutionResult:
    """Execute a circuit on the appropriate target.

    This is the main entry point for execution.
    """
    if target == ExecutionTarget.QUANTUM_HARDWARE:
        return execute_on_hardware(circuit, shots, parameter_values, backend_name)
    elif target == ExecutionTarget.QUANTUM_SIMULATE:
        return execute_on_simulator(circuit, shots, parameter_values)
    else:
        return ExecutionResult(
            counts={},
            shots=0,
            target=ExecutionTarget.CLASSICAL,
            most_likely="",
            success=False,
            error="Classical execution requested — no quantum circuit to run.",
        )
