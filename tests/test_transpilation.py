"""Tests for transpilation-based feasibility measurement.

Transpiling onto a 127-qubit heavy-hex map is slow, so these use the smallest
circuits that still exercise the behaviour, and rely on the module's lru_cache.
"""

import pytest

from quantum_agent.decision_engine import (
    HardwareConstraints,
    QuantumAlgorithm,
    decide,
    estimate_resources,
    extract_features,
)

transpilation = pytest.importorskip("quantum_agent.transpilation")


def test_default_backend_loads():
    backend = transpilation.get_backend()
    assert backend.num_qubits >= 27


def test_unknown_backend_raises():
    with pytest.raises(transpilation.BackendUnavailableError):
        transpilation.get_backend("fake_does_not_exist")


def test_vqe_estimate_is_close_to_compiled_reality():
    """A nearest-neighbour ladder maps well to hardware, so Eq 9 should hold up."""
    features = extract_features("quantum_simulation", 1024, num_variables=4)
    qubits, analytic = estimate_resources(QuantumAlgorithm.VQE, features)
    metrics = transpilation.measure(QuantumAlgorithm.VQE, qubits, analytic)
    assert metrics.depth_ratio < 3.0


def test_grover_estimate_badly_understates_compiled_depth():
    """Multi-controlled gates plus sparse connectivity: the estimate is far off."""
    features = extract_features("unstructured_search", 16, has_oracle=True)
    qubits, analytic = estimate_resources(QuantumAlgorithm.GROVER, features)
    metrics = transpilation.measure(QuantumAlgorithm.GROVER, qubits, analytic)
    assert metrics.depth > analytic * 10
    assert metrics.two_qubit_gates > 0


def test_measured_depth_changes_grover_routing():
    """The whole point: measuring depth must stop Grover reaching hardware.

    Uses N=64 rather than something smaller: below GROVER_MIN_SEARCH_SPACE the
    pipeline short-circuits at Stage 2 and never consults the feasibility gate.
    """
    kwargs = dict(problem_type="unstructured_search", search_space_size=64, has_oracle=True)

    analytic = decide(**kwargs, hardware=HardwareConstraints())
    measured = decide(**kwargs, hardware=HardwareConstraints(backend_name="fake_brisbane"))

    assert analytic.target.value == "quantum_hardware"
    assert measured.target.value == "quantum_simulate"
    assert measured.estimated_depth > analytic.estimated_depth
    # Stage 1 and 2 must be untouched by the Stage 3 backend choice.
    assert analytic.algorithm == measured.algorithm
    assert analytic.advantage == measured.advantage
    assert analytic.estimated_qubits == measured.estimated_qubits


def test_backend_name_defaults_to_analytic_path():
    """Omitting backend_name must reproduce the original behaviour exactly."""
    kwargs = dict(problem_type="unstructured_search", search_space_size=64, has_oracle=True)
    assert decide(**kwargs).estimated_depth == decide(
        **kwargs, hardware=HardwareConstraints()
    ).estimated_depth


def test_transpiled_depth_falls_back_on_failure():
    """A bad backend must not take down the pipeline."""
    assert transpilation.transpiled_depth(
        QuantumAlgorithm.GROVER, 4, 16, "fake_does_not_exist"
    ) == 16


def test_fidelity_is_a_probability_and_falls_with_circuit_size():
    """Deeper Grover circuits must have strictly worse estimated fidelity."""
    small = transpilation.measure(QuantumAlgorithm.GROVER, 3, 9)
    large = transpilation.measure(QuantumAlgorithm.GROVER, 6, 48)

    for m in (small, large):
        assert 0.0 <= m.estimated_fidelity <= 1.0
    assert large.estimated_fidelity < small.estimated_fidelity


def test_shallow_vqe_keeps_usable_fidelity():
    """VQE's shallow ansatz should stay far above Grover's vanishing fidelity."""
    vqe = transpilation.measure(QuantumAlgorithm.VQE, 4, 16)
    grover = transpilation.measure(QuantumAlgorithm.GROVER, 6, 48)
    assert vqe.estimated_fidelity > 0.5
    assert grover.estimated_fidelity < 0.01
