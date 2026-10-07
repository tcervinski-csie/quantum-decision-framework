"""Adversarial LLM Ablation Study.

Tests the LLM with deliberately misleading problem descriptions
that sound quantum but aren't, or sound classical but are quantum.

These are the problems a user might actually bring to an agent —
ambiguous, hype-laden, or deceptively framed descriptions.
"""

import json
import os
import time

import anthropic

from benchmarks.adversarial_problems import ADVERSARIAL_PROBLEMS
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

def query_bare_llm(client, description):
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Problem: {description}"}],
    )
    text = message.content[0].text.strip()
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return {"algorithm": "unknown", "advantage": "unknown", "reasoning": text}


def run_framework(description):
    hw = HardwareConstraints(max_qubits=0, max_circuit_depth=0)
    result = quantum_decision(description, hardware=hw)
    return {
        "algorithm": result["decision"]["algorithm"],
        "advantage": result["decision"]["advantage"],
    }


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY environment variable.")
        return

    client = anthropic.Anthropic(api_key=api_key)
    n = len(ADVERSARIAL_PROBLEMS)

    print("=" * 95)
    print("ADVERSARIAL ABLATION STUDY: Deliberately Misleading Problems")
    print("=" * 95)
    print(f"\nLLM: claude-haiku-4-5 (no tools)")
    print(f"Framework: quantum_decision (our pipeline)")
    print(f"Problems: {n} adversarial cases\n")

    print(f"{'ID':<30} {'Expected':<12} {'LLM':<12} {'LLM OK':<8} "
          f"{'FW':<12} {'FW OK':<8}")
    print("-" * 90)

    llm_correct = 0
    fw_correct = 0
    llm_failures = []
    fw_failures = []

    for p in ADVERSARIAL_PROBLEMS:
        llm_pred = query_bare_llm(client, p["description"])
        fw_pred = run_framework(p["description"])

        llm_algo_ok = llm_pred.get("algorithm", "") == p["expected_algorithm"]
        fw_algo_ok = fw_pred["algorithm"] == p["expected_algorithm"]

        llm_status = "PASS" if llm_algo_ok else "FAIL"
        fw_status = "PASS" if fw_algo_ok else "FAIL"

        if llm_algo_ok:
            llm_correct += 1
        else:
            llm_failures.append({
                "id": p["id"],
                "expected": p["expected_algorithm"],
                "got": llm_pred.get("algorithm", "?"),
                "reasoning": llm_pred.get("reasoning", ""),
                "trap": p["trap"],
            })

        if fw_algo_ok:
            fw_correct += 1
        else:
            fw_failures.append({
                "id": p["id"],
                "expected": p["expected_algorithm"],
                "got": fw_pred["algorithm"],
                "trap": p["trap"],
            })

        print(f"{p['id']:<30} {p['expected_algorithm']:<12} "
              f"{llm_pred.get('algorithm', '?'):<12} {llm_status:<8} "
              f"{fw_pred['algorithm']:<12} {fw_status:<8}")

        time.sleep(0.5)

    print(f"\n{'=' * 90}")
    print(f"\n--- Results ---")
    print(f"  Bare LLM:  {llm_correct}/{n} ({llm_correct/n*100:.0f}%)")
    print(f"  Framework: {fw_correct}/{n} ({fw_correct/n*100:.0f}%)")

    if llm_failures:
        print(f"\n--- LLM Failures ({len(llm_failures)}) ---")
        for f in llm_failures:
            print(f"\n  {f['id']}:")
            print(f"    Expected: {f['expected']}")
            print(f"    LLM said: {f['got']}")
            print(f"    LLM reasoning: {f['reasoning']}")
            print(f"    Trap: {f['trap']}")

    if fw_failures:
        print(f"\n--- Framework Failures ({len(fw_failures)}) ---")
        for f in fw_failures:
            print(f"\n  {f['id']}:")
            print(f"    Expected: {f['expected']}")
            print(f"    FW said:  {f['got']}")
            print(f"    Trap: {f['trap']}")

    # Categorize LLM errors
    if llm_failures:
        hype_tricked = sum(1 for f in llm_failures if f["id"].startswith("HYPE"))
        disguised_missed = sum(1 for f in llm_failures if f["id"].startswith("DISGUISED"))
        scale_missed = sum(1 for f in llm_failures if f["id"].startswith("SCALE"))
        misdir_followed = sum(1 for f in llm_failures if f["id"].startswith("MISDIR"))

        print(f"\n--- Error Categorization ---")
        if hype_tricked:
            print(f"  Fell for quantum hype: {hype_tricked}/5")
        if disguised_missed:
            print(f"  Missed disguised quantum problems: {disguised_missed}/3")
        if scale_missed:
            print(f"  Wrong on scale judgment: {scale_missed}/2")
        if misdir_followed:
            print(f"  Followed user's wrong suggestion: {misdir_followed}/2")


if __name__ == "__main__":
    main()
