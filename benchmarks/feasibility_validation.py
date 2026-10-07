"""Validate the Stage 3 feasibility gate against real compiled circuits.

Addresses IEEE Access reviewer 1 comment 9 — "these estimates... are not validated
against actual transpiled circuits... An improved feasibility model would include
information about the depth of the circuit, the gate count, and the execution
fidelity" — and reviewer 2 comment 4 / reviewer 1 comment 11 on Equation 9.

Method: for each supported algorithm, compare Equation 9's analytic estimate against
the circuit compiled for a real IBM device topology (fake_brisbane, 127-qubit Eagle,
heavy-hex, real basis gates and calibration). No credentials needed; these are
calibration snapshots of real hardware, though not live execution.

The headline is that the estimate's accuracy is ALGORITHM-DEPENDENT, which the
current single formula cannot express:

  * VQE   — a nearest-neighbour Ry/CNOT ladder maps almost 1:1 onto heavy-hex,
            so the analytic estimate is accurate and slightly conservative.
  * QAOA  — a ring graph needs modest SWAP routing; the estimate is off by a
            roughly constant ~7-8x.
  * Grover- multi-controlled X gates decompose into O(n^2)-ish two-qubit gates AND
            assume all-to-all connectivity, so the error is large and GROWS with
            problem size (20x -> 158x).

Usage:
    python -m benchmarks.feasibility_validation
"""

import argparse
import warnings

from quantum_agent.decision_engine import (
    HardwareConstraints,
    QuantumAlgorithm,
    check_feasibility,
    estimate_resources,
    extract_features,
)
from quantum_agent.transpilation import DEFAULT_BACKEND, measure

warnings.filterwarnings("ignore")

CASES = [
    ("unstructured_search", QuantumAlgorithm.GROVER, [8, 16, 32, 64, 128]),
    ("combinatorial_optimization", QuantumAlgorithm.QAOA, [4, 6, 8, 12, 16]),
    ("quantum_simulation", QuantumAlgorithm.VQE, [4, 6, 8, 12, 16]),
]


def features_for(problem_type: str, algorithm: QuantumAlgorithm, size: int):
    """Grover is sized by search space; QAOA and VQE by variable count."""
    if algorithm == QuantumAlgorithm.GROVER:
        return extract_features(problem_type, size, has_oracle=True)
    return extract_features(problem_type, 1024, num_variables=size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    args = parser.parse_args()

    limits = HardwareConstraints()
    print("=" * 86)
    print(f"STAGE 3 VALIDATION — Equation 9 vs. circuits compiled for {args.backend}")
    print("=" * 86)
    print(f"\nGate limits: <= {limits.max_qubits} qubits, <= {limits.max_circuit_depth} depth")
    print("Fidelity: product of per-gate success from the backend's own calibration,")
    print("including readout. Assumes independent errors and ignores crosstalk, so it")
    print("is an OPTIMISTIC UPPER BOUND on what the device could achieve.\n")

    changed = []
    for problem_type, algorithm, sizes in CASES:
        print(f"--- {algorithm.value.upper()} ---")
        print(
            f"{'size':>6} {'qubits':>7} {'Eq9':>6} {'real':>7} {'2q':>6} "
            f"{'ratio':>7} {'fidelity':>10}  {'Eq9 routes':>17}  {'real routes':>17}"
        )

        for size in sizes:
            features = features_for(problem_type, algorithm, size)
            qubits, analytic = estimate_resources(algorithm, features)
            metrics = measure(algorithm, qubits, analytic, args.backend)

            _, target_analytic = check_feasibility(qubits, analytic, limits)
            _, target_real = check_feasibility(qubits, metrics.depth, limits)

            flag = ""
            if target_analytic != target_real:
                flag = "  <-- CHANGES"
                changed.append((algorithm.value, size, target_analytic, target_real))

            print(
                f"{size:>6} {qubits:>7} {analytic:>6} {metrics.depth:>7} "
                f"{metrics.two_qubit_gates:>6} {metrics.depth_ratio:>6.1f}x "
                f"{metrics.estimated_fidelity:>10.2e}  "
                f"{target_analytic.value:>17}  {target_real.value:>17}{flag}"
            )
        print()

    print("=" * 86)
    print("ROUTING DECISIONS THAT CHANGE WHEN DEPTH IS MEASURED RATHER THAN ESTIMATED")
    print("=" * 86)
    if not changed:
        print("  none")
    else:
        for algorithm, size, was, now in changed:
            print(f"  {algorithm:>7} size={size:<5} {was.value} -> {now.value}")
        print(
            f"\n  {len(changed)} of the configurations above would have been dispatched to\n"
            "  hardware on the strength of an estimate the compiled circuit does not support."
        )


if __name__ == "__main__":
    main()
