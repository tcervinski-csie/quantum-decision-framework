"""Agent Zero custom tool: quantum_decision.

Drop this file into Agent Zero's `usr/tools/` directory.
It will be auto-discovered when the agent calls the "quantum_decision" tool.

Requires the quantum-agent package to be installed in the Agent Zero environment:
    pip install -e /path/to/quantum-ai
"""

import json

from python.helpers.tool import Tool, Response
from python.helpers.print_style import PrintStyle

from quantum_agent.agent_interface import quantum_decision as run_quantum_decision
from quantum_agent.decision_engine import HardwareConstraints


class QuantumDecision(Tool):

    async def execute(self, problem_description="", shots="1024",
                      parameter_values=None, **kwargs):

        if not problem_description:
            return Response(
                message="Error: problem_description is required.",
                break_loop=False,
            )

        shots = int(shots)

        # Parse parameter values if provided
        parsed_params = None
        if parameter_values:
            if isinstance(parameter_values, str):
                parsed_params = json.loads(parameter_values)
            elif isinstance(parameter_values, list):
                parsed_params = [float(v) for v in parameter_values]

        # Run the full quantum pipeline
        result = run_quantum_decision(
            problem_description=problem_description,
            shots=shots,
            parameter_values=parsed_params,
        )

        # Format response for the agent
        message = self._format_result(result)
        return Response(message=message, break_loop=False)

    def _format_result(self, result: dict) -> str:
        parts = []

        # Analysis
        a = result["analysis"]
        parts.append(
            f"Problem Analysis:\n"
            f"  Type: {a['problem_type']}\n"
            f"  Search space: {a['search_space_size']}\n"
            f"  Oracle: {a['has_oracle']}\n"
            f"  Confidence: {a['analyzer_confidence']:.0%}\n"
            f"  Reasoning: {a['analyzer_reasoning']}"
        )

        # Decision
        d = result["decision"]
        parts.append(
            f"\nDecision:\n"
            f"  Advantage: {d['advantage']}\n"
            f"  Algorithm: {d['algorithm']}\n"
            f"  Target: {d['target']}\n"
            f"  Qubits: {d['estimated_qubits']}\n"
            f"  Hardware feasible: {d['hardware_feasible']}\n"
            f"  Confidence: {d['confidence']:.0%}"
        )

        # Circuit
        if result["circuit_info"]:
            ci = result["circuit_info"]
            parts.append(
                f"\nCircuit:\n"
                f"  Qubits: {ci['num_qubits']}, Depth: {ci['depth']}, "
                f"  Gates: {ci['num_gates']}, Parameters: {ci['num_parameters']}"
            )

        # Execution
        if result["execution_result"]:
            er = result["execution_result"]
            if er["success"]:
                counts = er["counts"]
                top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
                top_str = "\n".join(
                    f"    |{s}>: {c} ({c/er['shots']*100:.1f}%)"
                    for s, c in top
                )
                parts.append(
                    f"\nExecution ({er['shots']} shots):\n"
                    f"  Most likely: |{er['most_likely']}>\n"
                    f"  Top results:\n{top_str}"
                )
            else:
                parts.append(f"\nExecution failed: {er['error']}")

        parts.append(f"\nSummary: {result['summary']}")
        return "\n".join(parts)
