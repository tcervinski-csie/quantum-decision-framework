"""Tests for the learned Stage 2 backend.

Encoding tests run unconditionally — they need numpy only. Tests that touch the
network skip cleanly when torch is absent or no checkpoint has been trained, so the
default install keeps a green suite without the optional dependency.
"""

import pytest

from quantum_agent.decision_engine import (
    GROVER_MIN_SEARCH_SPACE,
    ProblemType,
    QuantumAdvantage,
    QuantumAlgorithm,
    classify_advantage,
    decide,
    extract_features,
)
from quantum_agent.neural.encoding import (
    CLASSES,
    INPUT_DIM,
    OUTPUT_DIM,
    PROBLEM_TYPE_ORDER,
    decode_label,
    encode_features,
    label_index,
)


# --------------------------------------------------------------------------
# Encoding — no torch required
# --------------------------------------------------------------------------

def test_encoding_dimensions():
    features = extract_features("unstructured_search", 1024, has_oracle=True)
    assert len(encode_features(features)) == INPUT_DIM == 11


def test_encoding_one_hot_is_exclusive():
    features = extract_features("quantum_simulation", 256, num_variables=4)
    block = encode_features(features)[: len(PROBLEM_TYPE_ORDER)]
    assert sum(block) == 1.0


def test_encoding_is_deterministic():
    features = extract_features("combinatorial_optimization", 512, num_variables=8)
    assert encode_features(features) == encode_features(features)


def test_search_space_enters_logarithmically():
    """Doubling N must move the input by a constant step, not proportionally."""
    def size_component(n):
        return encode_features(extract_features("unstructured_search", n))[-2]

    step_a = size_component(256) - size_component(128)
    step_b = size_component(1024) - size_component(512)
    assert step_a == pytest.approx(step_b, abs=1e-9)


def test_label_roundtrip_covers_every_class():
    for index, (advantage, algorithm) in enumerate(CLASSES):
        assert label_index(advantage, algorithm) == index
        assert decode_label(index) == (advantage, algorithm)
    assert OUTPUT_DIM == 5


def test_every_reachable_rules_outcome_has_a_label():
    """Any (advantage, algorithm) the rules can emit must be encodable."""
    for problem_type in ProblemType:
        for oracle in (False, True):
            for n in (1, 63, 64, 10**6):
                features = extract_features(problem_type.value, n, has_oracle=oracle)
                advantage, algorithm, _ = classify_advantage(features)
                label_index(advantage, algorithm)  # raises KeyError if unreachable


# --------------------------------------------------------------------------
# Backend switching — no torch required
# --------------------------------------------------------------------------

def test_rules_backend_is_the_default():
    features = extract_features("unstructured_search", 1024, has_oracle=True)
    assert classify_advantage(features) == classify_advantage(features, backend="rules")


def test_unknown_backend_raises():
    features = extract_features("unstructured_search", 1024, has_oracle=True)
    with pytest.raises(ValueError, match="Unknown classification backend"):
        classify_advantage(features, backend="nonsense")


# --------------------------------------------------------------------------
# Network — skipped without torch or a trained checkpoint
# --------------------------------------------------------------------------

def _require_trained_model():
    pytest.importorskip("torch", reason="neural backend is an optional extra")
    from quantum_agent.neural.model import WEIGHTS_PATH

    if not WEIGHTS_PATH.exists():
        pytest.skip("no checkpoint; run python -m training.train_advantage_mlp")


def test_neural_backend_returns_the_rules_tuple_shape():
    _require_trained_model()
    features = extract_features("unstructured_search", 1024, has_oracle=True)
    advantage, algorithm, confidence = classify_advantage(features, backend="neural")

    assert isinstance(advantage, QuantumAdvantage)
    assert isinstance(algorithm, QuantumAlgorithm)
    assert 0.0 <= confidence <= 1.0


def test_neural_agrees_with_rules_across_the_grid():
    """The distilled network should reproduce its teacher nearly everywhere."""
    _require_trained_model()

    agree = total = 0
    for problem_type in ProblemType:
        for oracle in (False, True):
            for n in (1, 16, 100, 1000, 10**5):
                for num_vars in (0, 12):
                    features = extract_features(
                        problem_type.value, n, has_oracle=oracle, num_variables=num_vars
                    )
                    r_adv, r_alg, _ = classify_advantage(features)
                    n_adv, n_alg, _ = classify_advantage(features, backend="neural")
                    total += 1
                    agree += (r_adv, r_alg) == (n_adv, n_alg)

    assert agree / total >= 0.95, f"only {agree}/{total} agreed with the rules"


def test_confidence_is_lower_at_the_boundary_than_far_from_it():
    """The graded response is the whole point: uncertainty should peak at N=64."""
    _require_trained_model()

    def confidence_at(n):
        features = extract_features("unstructured_search", n, has_oracle=True)
        return classify_advantage(features, backend="neural")[2]

    assert confidence_at(GROVER_MIN_SEARCH_SPACE) < confidence_at(10**5)


def test_decide_threads_backend_through_but_keeps_stage_3_deterministic():
    """Stage 3 must not change with the Stage 2 backend."""
    _require_trained_model()

    kwargs = dict(problem_type="unstructured_search", search_space_size=1024, has_oracle=True)
    rules = decide(**kwargs)
    neural = decide(**kwargs, backend="neural")

    assert rules.algorithm == neural.algorithm
    assert (rules.estimated_qubits, rules.estimated_depth) == (
        neural.estimated_qubits,
        neural.estimated_depth,
    )
    assert rules.hardware_feasible == neural.hardware_feasible
