"""Empirical ground truth for the Grover advantage boundary.

Step 2 of the learned-backend evaluation. The rules assert a lower threshold
(GROVER_MIN_SEARCH_SPACE = 64, justified in a comment as "state preparation overhead
negates the quadratic speedup"), and the distilled network smooths that step to
N ~ 57. Deciding which is better needs a boundary derived from something other than
the rules themselves — otherwise the teacher grades its own student.

Metric: expected oracle queries to actually find the marked item.

    classical brute force : (N + 1) / 2          (expected, without replacement)
    Grover under noise    : k / p                 k = round(pi/4 * sqrt(N)) queries
                                                  per run, repeated until success

Grover is worth it exactly when k / p < (N + 1) / 2. This is measurable: p comes
from a noisy simulation, so the boundary is derived rather than asserted.

Caveat this cannot escape: simulated depolarizing noise is a model, not hardware.
It establishes where the advantage lies UNDER THAT MODEL. Real-backend validation
is a separate open item.

Usage:
    python -m benchmarks.advantage_ground_truth
"""

from math import ceil, pi, sqrt

from benchmarks.noise_analysis import (
    _make_decision,
    make_depolarizing_model,
    run_with_noise,
    success_probability,
)
from quantum_agent.decision_engine import (
    GROVER_MIN_SEARCH_SPACE,
    QuantumAlgorithm,
)
from quantum_agent.code_generator import _build_grover_circuit, optimal_grover_iterations

SHOTS = 2048
QUBIT_RANGE = range(2, 13)          # N = 4 .. 4096
NOISE_RATES = (0.0, 0.0005, 0.001, 0.005)


def grover_iterations(n_qubits: int) -> int:
    """Optimal Grover iteration count, delegating to the generator's exact formula."""
    return optimal_grover_iterations(2 ** n_qubits)


def classical_expected_queries(n: int) -> float:
    """Expected probes for classical brute force over N items."""
    return (n + 1) / 2


def measure(n_qubits: int, error_rate: float) -> dict:
    """Simulate Grover at this size/noise and return the query-cost comparison."""
    circuit = _build_grover_circuit(_make_decision(QuantumAlgorithm.GROVER, n_qubits))
    noise = make_depolarizing_model(error_rate) if error_rate > 0 else None

    counts = run_with_noise(circuit, noise, shots=SHOTS)
    p = success_probability(counts, "0" * n_qubits, SHOTS)

    n = 2 ** n_qubits
    k = grover_iterations(n_qubits)
    # p == 0 means never succeeds -> infinite expected cost
    quantum_cost = (k / p) if p > 0 else float("inf")
    classical_cost = classical_expected_queries(n)

    return {
        "n_qubits": n_qubits,
        "N": n,
        "error_rate": error_rate,
        "depth": circuit.depth(),
        "p_success": p,
        "iterations": k,
        "quantum_queries": quantum_cost,
        "classical_queries": classical_cost,
        "quantum_wins": quantum_cost < classical_cost,
    }


def main() -> None:
    print("=" * 78)
    print("EMPIRICAL GROVER ADVANTAGE BOUNDARY (expected oracle queries)")
    print("=" * 78)
    print(f"\nRules threshold under test: N >= {GROVER_MIN_SEARCH_SPACE}")
    print(f"Shots per point: {SHOTS}\n")

    results = []
    for rate in NOISE_RATES:
        label = "ideal" if rate == 0 else f"depolarizing {rate}"
        print(f"--- {label} ---")
        print(f"{'N':>7} {'depth':>6} {'p_succ':>7} {'q_queries':>11} {'c_queries':>10}  verdict")

        for n_qubits in QUBIT_RANGE:
            r = measure(n_qubits, rate)
            results.append(r)
            q = r["quantum_queries"]
            q_str = f"{q:11.1f}" if q != float("inf") else f"{'inf':>11}"
            verdict = "quantum" if r["quantum_wins"] else "CLASSICAL"
            print(
                f"{r['N']:>7} {r['depth']:>6} {r['p_success']:>7.3f} {q_str} "
                f"{r['classical_queries']:>10.1f}  {verdict}"
            )
        print()

    print("=" * 78)
    print("WHERE THE EMPIRICAL BOUNDARY FALLS")
    print("=" * 78)
    for rate in NOISE_RATES:
        subset = [r for r in results if r["error_rate"] == rate]
        winners = [r["N"] for r in subset if r["quantum_wins"]]
        label = "ideal" if rate == 0 else f"depolarizing {rate}"
        if winners:
            print(f"  {label:22s} quantum wins for N in [{min(winners)}, {max(winners)}]")
        else:
            print(f"  {label:22s} quantum never wins in the tested range")


if __name__ == "__main__":
    main()
