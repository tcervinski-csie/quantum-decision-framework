"""Agent Zero extension: inject quantum reasoning into the system prompt.

Drop this file into Agent Zero's `python/extensions/system_prompt/` directory.
It appends quantum computing awareness to the agent's system prompt.

Agent Zero discovers extensions by scanning for `Extension` subclasses, so this
must be a class — a module-level `execute()` function is silently ignored.
The hook receives the prompt list as the `system_prompt` keyword argument.
"""

from typing import Any

from python.helpers.extension import Extension
from agent import LoopData


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


class QuantumSystemPrompt(Extension):
    """Appends quantum-computing guidance to the agent's system prompt."""

    async def execute(
        self,
        system_prompt: list[str] = [],
        loop_data: LoopData = LoopData(),
        **kwargs: Any,
    ):
        system_prompt.append(QUANTUM_PROMPT)
