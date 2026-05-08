"""Ablation study: decision framework vs. no framework.

Runs all benchmark problems through:
  1. Agent WITH decision framework (our system)
  2. Baseline comparison against ground truth

Outputs a results table suitable for the paper's evaluation section.
"""

from quantum_agent.agent_interface import quantum_decision
from quantum_agent.decision_engine import HardwareConstraints
from benchmarks.benchmark_suite import BENCHMARK_PROBLEMS, BenchmarkProblem


def run_with_framework(problem: BenchmarkProblem) -> dict:
    """Run a problem through the full decision framework."""
    # Force simulator to avoid IBM credential requirement
    hw = HardwareConstraints(max_qubits=0, max_circuit_depth=0)

    result = quantum_decision(
        problem_description=problem.description,
        hardware=hw,
    )

    return {
        "predicted_algorithm": result["decision"]["algorithm"],
        "predicted_advantage": result["decision"]["advantage"],
        "analyzer_confidence": result["analysis"]["analyzer_confidence"],
        "decision_confidence": result["decision"]["confidence"],
        "problem_type_detected": result["analysis"]["problem_type"],
    }


def evaluate(problem: BenchmarkProblem, prediction: dict) -> dict:
    """Compare prediction against ground truth."""
    algo_correct = prediction["predicted_algorithm"] == problem.expected_algorithm
    advantage_correct = prediction["predicted_advantage"] == problem.expected_advantage

    # For the paper: a decision is "correct" if both algorithm and advantage match
    correct = algo_correct and advantage_correct

    return {
        "algorithm_correct": algo_correct,
        "advantage_correct": advantage_correct,
        "fully_correct": correct,
    }


def main():
    print("=" * 90)
    print("ABLATION STUDY: Quantum Decision Framework Accuracy")
    print("=" * 90)

    results = []
    categories = {}

    for problem in BENCHMARK_PROBLEMS:
        prediction = run_with_framework(problem)
        evaluation = evaluate(problem, prediction)

        results.append({
            "problem": problem,
            "prediction": prediction,
            "evaluation": evaluation,
        })

        cat = problem.category
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if evaluation["fully_correct"]:
            categories[cat]["correct"] += 1

    # Print detailed results
    print(f"\n{'ID':<25} {'Expected':<12} {'Predicted':<12} {'Adv Match':<12} {'Result':<8}")
    print("-" * 75)

    total_correct = 0
    for r in results:
        p = r["problem"]
        pred = r["prediction"]
        ev = r["evaluation"]
        status = "PASS" if ev["fully_correct"] else "FAIL"
        total_correct += 1 if ev["fully_correct"] else 0

        print(f"{p.id:<25} {p.expected_algorithm:<12} "
              f"{pred['predicted_algorithm']:<12} "
              f"{'yes' if ev['advantage_correct'] else 'NO':<12} "
              f"{status:<8}")

    # Summary
    accuracy = total_correct / len(results) * 100
    print(f"\n{'=' * 75}")
    print(f"Overall accuracy: {total_correct}/{len(results)} ({accuracy:.1f}%)")

    # Per-category breakdown
    print(f"\n--- Per-Category Accuracy ---")
    print(f"{'Category':<20} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 50)
    for cat, counts in sorted(categories.items()):
        cat_acc = counts["correct"] / counts["total"] * 100
        print(f"{cat:<20} {counts['correct']:<10} {counts['total']:<10} {cat_acc:.0f}%")

    # Failure analysis (for paper discussion)
    failures = [r for r in results if not r["evaluation"]["fully_correct"]]
    if failures:
        print(f"\n--- Failure Analysis ---")
        for r in failures:
            p = r["problem"]
            pred = r["prediction"]
            print(f"\n  {p.id}:")
            print(f"    Expected:  {p.expected_algorithm} ({p.expected_advantage})")
            print(f"    Got:       {pred['predicted_algorithm']} ({pred['predicted_advantage']})")
            print(f"    Detected:  type={pred['problem_type_detected']}, "
                  f"confidence={pred['analyzer_confidence']:.2f}")
            print(f"    Why wrong: {p.reasoning}")
    else:
        print(f"\n  No failures — all problems classified correctly.")

    return results


if __name__ == "__main__":
    main()
