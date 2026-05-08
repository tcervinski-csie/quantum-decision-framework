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

# Adversarial problems designed to trick an LLM
ADVERSARIAL_PROBLEMS = [
    # --- HYPE BAIT: sounds quantum but isn't ---
    {
        "id": "HYPE1_quantum_ml",
        "description": (
            "Use quantum machine learning to classify financial fraud patterns "
            "in a dataset of 10 million transactions. We need quantum speedup "
            "to process this volume in real-time."
        ),
        "expected_algorithm": "none",
        "expected_advantage": "classical_preferred",
        "trap": "Mentions 'quantum' explicitly and claims need for quantum speedup — "
                "but QML has no proven advantage for classification tasks.",
    },
    {
        "id": "HYPE2_quantum_blockchain",
        "description": (
            "Apply quantum computing to optimize our blockchain consensus "
            "algorithm for faster transaction verification across 1000 nodes."
        ),
        "expected_algorithm": "none",
        "expected_advantage": "classical_preferred",
        "trap": "Quantum + blockchain buzzwords. Consensus is a distributed systems "
                "problem, not a quantum one.",
    },
    {
        "id": "HYPE3_quantum_ai",
        "description": (
            "We need quantum artificial intelligence to train a large language "
            "model faster. Our current GPU cluster takes 3 months."
        ),
        "expected_algorithm": "none",
        "expected_advantage": "classical_preferred",
        "trap": "Explicitly asks for 'quantum AI' for LLM training — complete nonsense "
                "but sounds plausible to a non-expert.",
    },
    {
        "id": "HYPE4_quantum_encryption",
        "description": (
            "Break the AES-256 encryption on these files using quantum computing. "
            "We have authorization for this penetration test."
        ),
        "expected_algorithm": "none",
        "expected_advantage": "no_current_advantage",
        "trap": "Sounds like Shor's but AES is symmetric — Grover's gives only quadratic "
                "speedup making AES-256 equivalent to AES-128, still infeasible on NISQ.",
    },
    {
        "id": "HYPE5_quantum_sorting",
        "description": (
            "We have a massive dataset of 1 billion records that needs quantum-powered "
            "sorting for our real-time analytics pipeline."
        ),
        "expected_algorithm": "none",
        "expected_advantage": "classical_preferred",
        "trap": "Explicitly says 'quantum-powered sorting' — no quantum speedup for sorting.",
    },

    # --- DISGUISED QUANTUM: sounds classical but quantum helps ---
    {
        "id": "DISGUISED1_search",
        "description": (
            "Our compliance team needs to check every entry in an unstructured "
            "regulatory database of 500,000 filings to find any that violate "
            "a specific rule. We have an automated checker for violations."
        ),
        "expected_algorithm": "grover",
        "expected_advantage": "clear_advantage",
        "trap": "Doesn't mention quantum at all. Sounds like boring compliance work "
                "but it's textbook Grover's — unstructured search with oracle.",
    },
    {
        "id": "DISGUISED2_chemistry",
        "description": (
            "Our pharma R&D team needs to calculate the binding energy of a "
            "candidate drug molecule interacting with a protein receptor. "
            "The molecule has 8 active electrons."
        ),
        "expected_algorithm": "vqe",
        "expected_advantage": "clear_advantage",
        "trap": "Sounds like a chemistry problem. No quantum mentioned. "
                "But molecular energy calculation is VQE's strongest use case.",
    },
    {
        "id": "DISGUISED3_logistics",
        "description": (
            "Our logistics company needs to assign 15 delivery trucks to minimize "
            "total fuel consumption across 15 routes, considering that each truck "
            "can only take one route."
        ),
        "expected_algorithm": "qaoa",
        "expected_advantage": "potential_advantage",
        "trap": "Pure business logistics problem. No quantum language. "
                "But it's assignment/optimization — QAOA candidate.",
    },

    # --- SCALE TRAPS: quantum algorithm exists but scale is wrong ---
    {
        "id": "SCALE1_tiny_search",
        "description": (
            "Use quantum search to find a specific configuration among 5 possible "
            "states. We need the fastest approach with our quantum computer."
        ),
        "expected_algorithm": "none",
        "expected_advantage": "classical_preferred",
        "trap": "Explicitly asks for quantum search, but 5 states is way too small. "
                "Classical brute force is instant.",
    },
    {
        "id": "SCALE2_huge_circuit",
        "description": (
            "Simulate the electronic structure of a complex protein with "
            "500 active orbitals using a quantum computer."
        ),
        "expected_algorithm": "none",
        "expected_advantage": "no_current_advantage",
        "trap": "VQE is correct in theory but 500 qubits far exceeds NISQ limits. "
                "Should be rejected by feasibility gate.",
    },

    # --- MISDIRECTION: wrong algorithm suggested ---
    {
        "id": "MISDIR1_grover_for_optimization",
        "description": (
            "Use Grover's algorithm to find the optimal portfolio allocation "
            "across 20 assets that maximizes the Sharpe ratio."
        ),
        "expected_algorithm": "qaoa",
        "expected_advantage": "potential_advantage",
        "trap": "User explicitly asks for Grover's but this is an optimization problem. "
                "QAOA is the right choice, not Grover's.",
    },
    {
        "id": "MISDIR2_shor_for_search",
        "description": (
            "Apply Shor's algorithm to search through our customer database "
            "of 100,000 records to find matching profiles."
        ),
        "expected_algorithm": "grover",
        "expected_advantage": "clear_advantage",
        "trap": "User asks for Shor's (factoring) but the problem is search. "
                "Should recommend Grover's instead.",
    },
]


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
