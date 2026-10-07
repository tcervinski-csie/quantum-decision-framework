"""Adversarial benchmark problems (data only).

Separated from `llm_ablation_adversarial.py` so the dataset can be loaded without
the `anthropic` package or an API key. The LLM ablation needs both; the dataset
itself is plain data, and the ML ablation and any reproduction of the evaluation
must be able to import it standalone.

Each entry: id, description, expected_algorithm, expected_advantage, trap
(the trap field documents WHY the problem is adversarial).
"""

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
