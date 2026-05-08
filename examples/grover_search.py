"""Example: Grover's search on a 10-qubit space (1024 elements).

Demonstrates the full pipeline:
1. Decision engine classifies as clear quantum advantage
2. Code generator builds a Grover circuit
3. Executor runs on Aer simulator
4. Result shows the marked state found with high probability
"""

from quantum_agent.agent_tool import quantum_tool


def main():
    print("=" * 60)
    print("Grover's Search — Unstructured Search over 1024 elements")
    print("=" * 60)

    result = quantum_tool(
        problem_type="unstructured_search",
        search_space_size=1024,
        has_oracle=True,
        shots=2048,
    )

    decision = result["decision"]
    print(f"\n--- Decision ---")
    print(f"  Advantage:  {decision['advantage']}")
    print(f"  Algorithm:  {decision['algorithm']}")
    print(f"  Target:     {decision['target']}")
    print(f"  Qubits:     {decision['estimated_qubits']}")
    print(f"  Confidence: {decision['confidence']}")

    circuit = result["circuit_info"]
    print(f"\n--- Circuit ---")
    print(f"  Qubits:     {circuit['num_qubits']}")
    print(f"  Depth:      {circuit['depth']}")
    print(f"  Gates:      {circuit['num_gates']}")

    execution = result["execution_result"]
    print(f"\n--- Execution ---")
    print(f"  Success:      {execution['success']}")
    print(f"  Shots:        {execution['shots']}")
    print(f"  Most likely:  {execution['most_likely']}")

    # Show top 5 results
    counts = execution["counts"]
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Top 5 outcomes:")
    for state, count in sorted_counts:
        pct = count / execution["shots"] * 100
        print(f"    |{state}> : {count} ({pct:.1f}%)")

    print(f"\n--- Summary ---")
    print(f"  {result['summary']}")


if __name__ == "__main__":
    main()
