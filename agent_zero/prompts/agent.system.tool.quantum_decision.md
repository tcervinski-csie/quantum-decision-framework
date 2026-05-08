### quantum_decision:
Analyze a computational problem to determine if quantum computing provides an advantage. If quantum is appropriate, generates and executes the quantum circuit automatically. **Always call this tool before deciding whether to use a quantum or classical approach.** Do not guess — let the tool's decision framework classify the problem.

**When to use:**
- User presents any computational problem that *might* benefit from quantum computing
- Unstructured search, combinatorial optimization, quantum simulation problems
- When you need to determine if quantum is appropriate for a given task

**What it returns:**
- Problem analysis (type, oracle, structure)
- Quantum advantage classification (clear / potential / none / classical preferred)
- Circuit generation and execution results (if quantum was selected)
- Human-readable summary with reasoning

**usage:**
~~~json
{
    "tool_name": "quantum_decision",
    "tool_args": {
        "problem_description": "Search through an unsorted database of 100000 elements to find one matching a verification condition.",
        "shots": 1024,
        "parameter_values": [0.5, 0.5]
    }
}
~~~

**Arguments:**
- `problem_description` (required): Natural language description of the problem
- `shots` (optional, default 1024): Number of quantum measurement shots
- `parameter_values` (optional): Float array for variational circuits (QAOA needs [gamma, beta], VQE needs theta values)
