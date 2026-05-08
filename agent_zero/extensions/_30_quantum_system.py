"""Agent Zero extension: inject quantum reasoning into the system prompt.

Drop this file into Agent Zero's `python/extensions/system_prompt/` directory.
It appends quantum computing awareness to the agent's system prompt.
"""

import os

QUANTUM_PROMPT = """

## Quantum Computing Awareness

You have access to a quantum computing decision framework via the `quantum_decision` tool.
When a user presents a computational problem, consider whether quantum computing might help.

**Always use the tool to decide — do not rely on your own judgment about quantum advantage.**

Problems where quantum may help:
- Unstructured search with a verifiable condition (Grover's algorithm)
- Combinatorial optimization: MaxCut, TSP, portfolio, scheduling (QAOA)
- Quantum system simulation: molecular energy, spin chains (VQE)

Problems where quantum does NOT help today:
- Integer factoring / RSA (Shor's needs fault-tolerant hardware we don't have)
- Linear systems (HHL has input/output bottleneck)
- Machine learning (no proven quantum advantage)
- Small search spaces (< 64 elements — classical is faster)

You are in the NISQ era: max 127 qubits, shallow circuits only.
Never claim quantum speedup without the decision tool confirming it.
"""


async def execute(agent, prompts: list, **kwargs):
    """Append quantum reasoning context to the system prompt."""
    prompts.append(QUANTUM_PROMPT)
    return prompts
