"""Example: QAOA for MaxCut on a 6-node ring graph with parameter optimization.

MaxCut: partition graph nodes into two sets to maximize edges between them.
A ring graph with 6 nodes has optimal MaxCut = 6 (alternating partition).

Demonstrates:
1. Decision engine classifies as potential quantum advantage
2. Code generator builds a QAOA circuit with parameterized layers
3. Classical optimizer tunes (gamma, beta) over repeated simulator runs
4. Convergence from random cut values toward the optimal solution
"""

from math import pi

import numpy as np
from scipy.optimize import minimize

from quantum_agent.decision_engine import decide, ExecutionTarget
from quantum_agent.code_generator import generate_circuit
from quantum_agent.executor import execute_on_simulator


def evaluate_maxcut(bitstring: str, edges: list[tuple[int, int]]) -> int:
    """Count how many edges are cut by a given partition."""
    return sum(1 for i, j in edges if bitstring[i] != bitstring[j])


def expected_cut_value(counts: dict[str, int], edges: list[tuple[int, int]], shots: int) -> float:
    """Compute the expected MaxCut value from measurement counts."""
    total = 0.0
    for bitstring, count in counts.items():
        total += evaluate_maxcut(bitstring, edges) * count
    return total / shots


def main():
    n_nodes = 6
    edges = [(i, (i + 1) % n_nodes) for i in range(n_nodes)]
    shots = 2048

    print("=" * 60)
    print(f"QAOA MaxCut — {n_nodes}-node ring graph with optimization")
    print(f"Edges: {edges}")
    print(f"Optimal MaxCut value: {n_nodes} (alternating partition)")
    print("=" * 60)

    # --- Step 1: Decision ---
    decision = decide(
        problem_type="combinatorial_optimization",
        search_space_size=2**n_nodes,
        num_variables=n_nodes,
    )
    print(f"\n--- Decision ---")
    print(f"  Advantage:  {decision.advantage.value}")
    print(f"  Algorithm:  {decision.algorithm.value}")
    print(f"  Qubits:     {decision.estimated_qubits}")

    # --- Step 2: Build circuit (once) ---
    circuit = generate_circuit(decision)
    print(f"\n--- Circuit ---")
    print(f"  Qubits:     {circuit.num_qubits}")
    print(f"  Depth:      {circuit.depth()}")
    print(f"  Parameters: {len(circuit.parameters)} (γ, β)")

    # --- Step 3: Optimization loop ---
    print(f"\n--- Optimization ---")
    iteration = [0]

    def cost_function(params):
        gamma, beta = params
        result = execute_on_simulator(circuit, shots=shots, parameter_values=[gamma, beta])
        if not result.success:
            return 0.0  # worst case
        # Negate because we maximize cut but scipy minimizes
        cost = -expected_cut_value(result.counts, edges, shots)
        iteration[0] += 1
        if iteration[0] % 5 == 0 or iteration[0] == 1:
            print(f"  Iter {iteration[0]:3d}: γ={gamma:.4f}, β={beta:.4f}, "
                  f"expected cut = {-cost:.3f}/{n_nodes}")
        return cost

    # Start from a reasonable initial point
    x0 = np.array([pi / 4, pi / 8])

    opt_result = minimize(
        cost_function,
        x0,
        method="COBYLA",
        options={"maxiter": 60, "rhobeg": 0.5},
    )

    best_gamma, best_beta = opt_result.x
    print(f"\n  Optimized: γ={best_gamma:.4f}, β={best_beta:.4f}")
    print(f"  Optimizer iterations: {opt_result.nfev}")

    # --- Step 4: Final run with optimized parameters ---
    print(f"\n--- Final Execution (optimized parameters) ---")
    final_result = execute_on_simulator(
        circuit, shots=shots * 4, parameter_values=[best_gamma, best_beta]
    )

    final_expected = expected_cut_value(final_result.counts, edges, final_result.shots)
    print(f"  Shots:          {final_result.shots}")
    print(f"  Expected cut:   {final_expected:.3f}/{n_nodes}")
    print(f"  Most likely:    {final_result.most_likely}")

    # Show top results
    sorted_counts = sorted(final_result.counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Top 10 outcomes:")
    for bitstring, count in sorted_counts:
        cut_value = evaluate_maxcut(bitstring, edges)
        pct = count / final_result.shots * 100
        marker = " ← optimal!" if cut_value == n_nodes else ""
        print(f"    |{bitstring}> : {count:4d} ({pct:5.1f}%) — cut = {cut_value}/{n_nodes}{marker}")

    # --- Comparison ---
    print(f"\n--- Comparison ---")
    # Random baseline: expected cut of a random partition on a ring
    random_expected = len(edges) / 2  # each edge has 50% chance of being cut
    print(f"  Random baseline: {random_expected:.1f}/{n_nodes}")
    print(f"  QAOA (p=1):      {final_expected:.3f}/{n_nodes}")
    print(f"  Optimal:         {n_nodes}/{n_nodes}")
    approx_ratio = final_expected / n_nodes
    print(f"  Approximation ratio: {approx_ratio:.3f}")


if __name__ == "__main__":
    main()
