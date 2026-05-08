"""Expanded QAOA evaluation across multiple graph types and sizes.

Tests QAOA MaxCut performance on:
- Graph types: ring, complete, random (Erdős–Rényi)
- Sizes: 4, 6, 8, 10 nodes
- QAOA layers: p=1 vs p=2

Outputs a results table with approximation ratios for the paper.
"""

import random
from itertools import combinations
from math import pi

import numpy as np
from scipy.optimize import minimize

from quantum_agent.decision_engine import decide, HardwareConstraints
from quantum_agent.code_generator import generate_circuit, _build_qaoa_circuit
from quantum_agent.decision_engine import (
    Decision,
    ExecutionTarget,
    QuantumAdvantage,
    QuantumAlgorithm,
)
from quantum_agent.executor import execute_on_simulator


def maxcut_value(bitstring: str, edges: list[tuple[int, int]]) -> int:
    """Count edges cut by a partition."""
    return sum(1 for i, j in edges if bitstring[i] != bitstring[j])


def expected_cut(counts: dict[str, int], edges: list[tuple[int, int]], shots: int) -> float:
    """Compute expected cut value from measurement counts."""
    return sum(maxcut_value(bs, edges) * c for bs, c in counts.items()) / shots


def brute_force_maxcut(n: int, edges: list[tuple[int, int]]) -> int:
    """Find optimal MaxCut value by exhaustive search."""
    best = 0
    for i in range(2 ** n):
        bs = format(i, f"0{n}b")
        best = max(best, maxcut_value(bs, edges))
    return best


# --- Graph generators ---

def ring_graph(n: int) -> list[tuple[int, int]]:
    return [(i, (i + 1) % n) for i in range(n)]


def complete_graph(n: int) -> list[tuple[int, int]]:
    return list(combinations(range(n), 2))


def random_graph(n: int, p: float = 0.5, seed: int = 42) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    # Ensure connected: add ring edges if needed
    for i in range(n):
        edge = (i, (i + 1) % n)
        if edge not in edges:
            edges.append(edge)
    return edges


def optimize_qaoa(n_qubits: int, edges: list[tuple[int, int]], p: int = 1, shots: int = 1024) -> dict:
    """Run QAOA with classical parameter optimization.

    Args:
        n_qubits: Number of graph nodes / qubits.
        edges: Graph edges.
        p: Number of QAOA layers.
        shots: Shots per circuit evaluation.

    Returns:
        Dict with best_params, best_cut, approximation_ratio, etc.
    """
    # Build decision and circuit
    decision = Decision(
        advantage=QuantumAdvantage.POTENTIAL,
        algorithm=QuantumAlgorithm.QAOA,
        target=ExecutionTarget.QUANTUM_SIMULATE,
        estimated_qubits=n_qubits,
        estimated_depth=2 * n_qubits * p,
        confidence=0.6,
        reasoning="QAOA evaluation",
        hardware_feasible=False,
    )

    circuit = _build_qaoa_circuit(decision, edges=edges, p=p)
    n_params = 2 * p  # gamma + beta per layer

    optimal_cut = brute_force_maxcut(n_qubits, edges)

    def cost_fn(params):
        result = execute_on_simulator(circuit, shots=shots, parameter_values=list(params))
        if not result.success:
            return 0.0
        return -expected_cut(result.counts, edges, shots)

    # Multiple random starts to avoid local minima
    best_cost = 0.0
    best_params = None
    best_counts = None

    for trial in range(3):
        x0 = np.random.RandomState(seed=trial).uniform(0, pi, size=n_params)
        opt = minimize(cost_fn, x0, method="COBYLA", options={"maxiter": 50, "rhobeg": 0.5})

        if -opt.fun > best_cost:
            best_cost = -opt.fun
            best_params = opt.x

    # Final high-shot run with best parameters
    final = execute_on_simulator(circuit, shots=shots * 4, parameter_values=list(best_params))
    final_cut = expected_cut(final.counts, edges, final.shots)
    approx_ratio = final_cut / optimal_cut if optimal_cut > 0 else 0.0

    return {
        "n_qubits": n_qubits,
        "n_edges": len(edges),
        "optimal_cut": optimal_cut,
        "qaoa_cut": round(final_cut, 3),
        "approx_ratio": round(approx_ratio, 3),
        "best_params": [round(p, 4) for p in best_params],
        "p_layers": p,
    }


def main():
    np.random.seed(42)

    graph_types = {
        "Ring": ring_graph,
        "Complete": complete_graph,
        "Random(0.5)": lambda n: random_graph(n, p=0.5),
    }
    sizes = [4, 6, 8]
    p_values = [1, 2]

    print("=" * 90)
    print("QAOA MaxCut Evaluation — Multiple Graphs, Sizes, and Depths")
    print("=" * 90)

    # Results table
    print(f"\n{'Graph':<15} {'n':<5} {'Edges':<7} {'p':<5} {'Optimal':<9} "
          f"{'QAOA':<9} {'Ratio':<9} {'Random':<9}")
    print("-" * 80)

    all_results = []

    for graph_name, graph_fn in graph_types.items():
        for n in sizes:
            edges = graph_fn(n)
            random_baseline = len(edges) / 2  # expected cut from random partition

            for p in p_values:
                print(f"  Running: {graph_name} n={n} p={p}...", end=" ", flush=True)
                result = optimize_qaoa(n, edges, p=p, shots=512)
                result["graph_type"] = graph_name
                result["random_baseline"] = round(random_baseline, 1)
                all_results.append(result)

                print(f"ratio={result['approx_ratio']:.3f}")

                print(f"{graph_name:<15} {n:<5} {len(edges):<7} {p:<5} "
                      f"{result['optimal_cut']:<9} {result['qaoa_cut']:<9} "
                      f"{result['approx_ratio']:<9} {random_baseline:<9.1f}")

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("\n--- Summary ---")

    for p in p_values:
        p_results = [r for r in all_results if r["p_layers"] == p]
        avg_ratio = sum(r["approx_ratio"] for r in p_results) / len(p_results)
        min_ratio = min(r["approx_ratio"] for r in p_results)
        max_ratio = max(r["approx_ratio"] for r in p_results)
        print(f"  QAOA p={p}: avg ratio={avg_ratio:.3f}, "
              f"min={min_ratio:.3f}, max={max_ratio:.3f}")

    # p=2 vs p=1 improvement
    p1_results = {(r["graph_type"], r["n_qubits"]): r["approx_ratio"]
                  for r in all_results if r["p_layers"] == 1}
    p2_results = {(r["graph_type"], r["n_qubits"]): r["approx_ratio"]
                  for r in all_results if r["p_layers"] == 2}

    improvements = []
    for key in p1_results:
        if key in p2_results:
            improvements.append(p2_results[key] - p1_results[key])

    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"\n  Average p=2 improvement over p=1: {avg_improvement:+.3f}")


if __name__ == "__main__":
    main()
