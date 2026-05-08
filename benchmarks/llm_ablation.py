"""LLM Ablation Study: bare LLM vs. LLM + Decision Framework.

Sends each benchmark problem to a raw LLM (no quantum tools) and asks
it to decide whether quantum computing is appropriate and which algorithm
to use. Compares the LLM's answers against ground truth and against
our decision framework's answers.

Usage:
    ANTHROPIC_API_KEY=sk-... python3 -m benchmarks.llm_ablation
"""

import json
import os
import time

import anthropic

from benchmarks.benchmark_suite import BENCHMARK_PROBLEMS, BenchmarkProblem
from quantum_agent.agent_interface import quantum_decision
from quantum_agent.decision_engine import HardwareConstraints


SYSTEM_PROMPT = """\
You are a quantum computing expert. Given a computational problem, decide:
1. Should this problem be solved using quantum computing or classical computing?
2. If quantum, which algorithm? Choose from: grover, qaoa, vqe, or none (classical).
3. What is the quantum advantage level? Choose from: clear_advantage, potential_advantage, no_current_advantage, classical_preferred.

Consider NISQ-era constraints: max 127 qubits, noisy gates, shallow circuits only.
Algorithms requiring fault-tolerant quantum computing (Shor's, HHL) are NOT feasible today.

Respond ONLY with valid JSON, no other text:
{"algorithm": "grover|qaoa|vqe|none", "advantage": "clear_advantage|potential_advantage|no_current_advantage|classical_preferred", "reasoning": "one sentence explanation"}
"""


def query_bare_llm(client: anthropic.Anthropic, problem: str) -> dict:
    """Ask the LLM directly without any tools."""
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Problem: {problem}"}],
    )

    text = message.content[0].text.strip()

    # Parse JSON response
    try:
        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return {"algorithm": "unknown", "advantage": "unknown", "reasoning": text}


def run_framework(problem: BenchmarkProblem) -> dict:
    """Run problem through our decision framework."""
    hw = HardwareConstraints(max_qubits=0, max_circuit_depth=0)
    result = quantum_decision(problem.description, hardware=hw)
    return {
        "algorithm": result["decision"]["algorithm"],
        "advantage": result["decision"]["advantage"],
    }


def evaluate(predicted: dict, expected: BenchmarkProblem) -> dict:
    algo_correct = predicted.get("algorithm", "") == expected.expected_algorithm
    adv_correct = predicted.get("advantage", "") == expected.expected_advantage
    return {
        "algorithm_correct": algo_correct,
        "advantage_correct": adv_correct,
        "fully_correct": algo_correct and adv_correct,
    }


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY environment variable.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    print("=" * 95)
    print("LLM ABLATION STUDY: Bare LLM vs. Decision Framework")
    print("=" * 95)
    print(f"\nLLM: claude-haiku-4-5 (no tools, just prompted)")
    print(f"Framework: quantum_decision (our 3-stage pipeline)")
    print(f"Benchmark: {len(BENCHMARK_PROBLEMS)} problems with ground truth\n")

    llm_results = []
    fw_results = []

    print(f"{'ID':<25} {'Expected':<12} {'LLM':<12} {'LLM OK':<8} "
          f"{'Framework':<12} {'FW OK':<8}")
    print("-" * 85)

    for problem in BENCHMARK_PROBLEMS:
        # Query bare LLM
        llm_pred = query_bare_llm(client, problem.description)
        llm_eval = evaluate(llm_pred, problem)
        llm_results.append({"problem": problem, "prediction": llm_pred, "eval": llm_eval})

        # Query framework
        fw_pred = run_framework(problem)
        fw_eval = evaluate(fw_pred, problem)
        fw_results.append({"problem": problem, "prediction": fw_pred, "eval": fw_eval})

        llm_status = "PASS" if llm_eval["fully_correct"] else "FAIL"
        fw_status = "PASS" if fw_eval["fully_correct"] else "FAIL"

        print(f"{problem.id:<25} {problem.expected_algorithm:<12} "
              f"{llm_pred.get('algorithm', '?'):<12} {llm_status:<8} "
              f"{fw_pred['algorithm']:<12} {fw_status:<8}")

        # Rate limit
        time.sleep(0.5)

    # Summary
    llm_correct = sum(1 for r in llm_results if r["eval"]["fully_correct"])
    fw_correct = sum(1 for r in fw_results if r["eval"]["fully_correct"])
    llm_algo_correct = sum(1 for r in llm_results if r["eval"]["algorithm_correct"])
    fw_algo_correct = sum(1 for r in fw_results if r["eval"]["algorithm_correct"])

    n = len(BENCHMARK_PROBLEMS)

    print(f"\n{'=' * 85}")
    print(f"\n--- Overall Results ---")
    print(f"{'Metric':<30} {'Bare LLM':<20} {'Framework':<20}")
    print(f"-" * 70)
    print(f"{'Algorithm correct':<30} {llm_algo_correct}/{n} ({llm_algo_correct/n*100:.0f}%)"
          f"{'':>5} {fw_algo_correct}/{n} ({fw_algo_correct/n*100:.0f}%)")
    print(f"{'Fully correct (algo+adv)':<30} {llm_correct}/{n} ({llm_correct/n*100:.0f}%)"
          f"{'':>5} {fw_correct}/{n} ({fw_correct/n*100:.0f}%)")

    # Category breakdown
    categories = {}
    for r_llm, r_fw in zip(llm_results, fw_results):
        cat = r_llm["problem"].category
        if cat not in categories:
            categories[cat] = {"llm_correct": 0, "fw_correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r_llm["eval"]["fully_correct"]:
            categories[cat]["llm_correct"] += 1
        if r_fw["eval"]["fully_correct"]:
            categories[cat]["fw_correct"] += 1

    print(f"\n--- Per-Category Breakdown ---")
    print(f"{'Category':<20} {'LLM':<15} {'Framework':<15}")
    print("-" * 50)
    for cat, counts in sorted(categories.items()):
        llm_pct = counts["llm_correct"] / counts["total"] * 100
        fw_pct = counts["fw_correct"] / counts["total"] * 100
        print(f"{cat:<20} {counts['llm_correct']}/{counts['total']} ({llm_pct:.0f}%)"
              f"{'':>5} {counts['fw_correct']}/{counts['total']} ({fw_pct:.0f}%)")

    # LLM failure analysis
    llm_failures = [r for r in llm_results if not r["eval"]["fully_correct"]]
    if llm_failures:
        print(f"\n--- LLM Failure Analysis ---")
        for r in llm_failures:
            p = r["problem"]
            pred = r["prediction"]
            print(f"\n  {p.id}:")
            print(f"    Expected:  {p.expected_algorithm} ({p.expected_advantage})")
            print(f"    LLM said:  {pred.get('algorithm', '?')} ({pred.get('advantage', '?')})")
            print(f"    LLM reason: {pred.get('reasoning', 'N/A')}")
            print(f"    Why wrong: {p.reasoning}")

    # Key finding for the paper
    improvement = fw_correct - llm_correct
    print(f"\n{'=' * 85}")
    print(f"KEY FINDING: Decision framework improves accuracy by "
          f"+{improvement} problems ({improvement/n*100:.0f} percentage points)")
    print(f"  Bare LLM:  {llm_correct}/{n} ({llm_correct/n*100:.0f}%)")
    print(f"  Framework: {fw_correct}/{n} ({fw_correct/n*100:.0f}%)")

    if llm_failures:
        print(f"\nCommon LLM errors:")
        # Categorize errors
        hallucinated_quantum = sum(
            1 for r in llm_failures
            if r["prediction"].get("algorithm", "none") != "none"
            and r["problem"].expected_algorithm == "none"
        )
        missed_quantum = sum(
            1 for r in llm_failures
            if r["prediction"].get("algorithm", "none") == "none"
            and r["problem"].expected_algorithm != "none"
        )
        wrong_algorithm = sum(
            1 for r in llm_failures
            if r["prediction"].get("algorithm", "none") != "none"
            and r["problem"].expected_algorithm != "none"
            and r["prediction"].get("algorithm") != r["problem"].expected_algorithm
        )
        wrong_advantage = sum(
            1 for r in llm_failures
            if r["eval"]["algorithm_correct"] and not r["eval"]["advantage_correct"]
        )

        if hallucinated_quantum:
            print(f"  - Hallucinated quantum advantage: {hallucinated_quantum} "
                  f"(suggested quantum when classical is correct)")
        if missed_quantum:
            print(f"  - Missed quantum opportunity: {missed_quantum} "
                  f"(said classical when quantum applies)")
        if wrong_algorithm:
            print(f"  - Wrong algorithm: {wrong_algorithm} "
                  f"(picked quantum but wrong algorithm)")
        if wrong_advantage:
            print(f"  - Wrong advantage level: {wrong_advantage} "
                  f"(right algorithm, wrong classification)")


if __name__ == "__main__":
    main()
