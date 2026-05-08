"""Ablation study benchmark suite.

15 problems with known ground-truth labels for whether quantum computing
is appropriate. Used to compare:
  - Agent WITH decision framework (structured classification)
  - Agent WITHOUT decision framework (LLM-only reasoning)

Each problem has:
  - description: natural language problem statement
  - expected_decision: correct classification (quantum algorithm or "classical")
  - expected_advantage: the advantage level
  - reasoning: why this is the correct answer (for paper discussion)
"""

from dataclasses import dataclass


@dataclass
class BenchmarkProblem:
    id: str
    description: str
    expected_algorithm: str  # "grover", "qaoa", "vqe", "none"
    expected_advantage: str  # "clear_advantage", "potential_advantage", "no_current_advantage", "classical_preferred"
    category: str  # for grouping in results table
    reasoning: str


BENCHMARK_PROBLEMS = [
    # === TRUE POSITIVES: Quantum should be selected ===

    BenchmarkProblem(
        id="TP1_grover_database",
        description="Search through an unsorted database of 1000000 elements to find "
                    "the one record that satisfies a specific validation condition.",
        expected_algorithm="grover",
        expected_advantage="clear_advantage",
        category="search",
        reasoning="Large unstructured search with verifiable oracle — textbook Grover's.",
    ),
    BenchmarkProblem(
        id="TP2_grover_sat",
        description="Find a satisfying assignment for a boolean constraint satisfaction "
                    "problem with 16 variables where we can verify any candidate solution.",
        expected_algorithm="grover",
        expected_advantage="clear_advantage",
        category="search",
        reasoning="SAT with oracle (verifier) maps directly to Grover's.",
    ),
    BenchmarkProblem(
        id="TP3_qaoa_maxcut",
        description="Find the maximum cut partition of a graph with 8 nodes "
                    "to maximize the number of edges between the two sets.",
        expected_algorithm="qaoa",
        expected_advantage="potential_advantage",
        category="optimization",
        reasoning="MaxCut is the canonical QAOA problem.",
    ),
    BenchmarkProblem(
        id="TP4_qaoa_portfolio",
        description="Optimize a portfolio of 10 assets to find the allocation that "
                    "minimizes risk while maximizing expected return.",
        expected_algorithm="qaoa",
        expected_advantage="potential_advantage",
        category="optimization",
        reasoning="Portfolio optimization maps to quadratic binary optimization (QAOA).",
    ),
    BenchmarkProblem(
        id="TP5_vqe_molecule",
        description="Simulate the ground state energy of a lithium hydride molecule "
                    "using 6 qubits to determine its chemical properties.",
        expected_algorithm="vqe",
        expected_advantage="clear_advantage",
        category="simulation",
        reasoning="Molecular simulation is the strongest NISQ use case for VQE.",
    ),
    BenchmarkProblem(
        id="TP6_vqe_spin",
        description="Find the ground state of a Hamiltonian describing a 4 spin chain "
                    "with nearest-neighbor interactions.",
        expected_algorithm="vqe",
        expected_advantage="clear_advantage",
        category="simulation",
        reasoning="Spin chain Hamiltonian is a natural VQE problem.",
    ),

    # === TRUE NEGATIVES: Classical should be selected ===

    BenchmarkProblem(
        id="TN1_ml_classification",
        description="Train a neural network to classify images of cats and dogs "
                    "from a dataset of 50000 labeled examples.",
        expected_algorithm="none",
        expected_advantage="classical_preferred",
        category="ml",
        reasoning="No proven quantum advantage for image classification.",
    ),
    BenchmarkProblem(
        id="TN2_ml_regression",
        description="Build a regression model to predict house prices based on "
                    "15 features using historical sales data.",
        expected_algorithm="none",
        expected_advantage="classical_preferred",
        category="ml",
        reasoning="Standard ML regression — no quantum benefit.",
    ),
    BenchmarkProblem(
        id="TN3_small_search",
        description="Search through a list of 20 elements to find one that "
                    "satisfies a given condition.",
        expected_algorithm="none",
        expected_advantage="classical_preferred",
        category="search",
        reasoning="Search space too small (20 < 64) — classical is faster.",
    ),
    BenchmarkProblem(
        id="TN4_sorting",
        description="Sort an array of 10000 integers in ascending order.",
        expected_algorithm="none",
        expected_advantage="classical_preferred",
        category="classical",
        reasoning="Sorting has no quantum speedup — O(n log n) is optimal.",
    ),

    # === TRAPS: Problems that sound quantum but aren't (NISQ-era) ===

    BenchmarkProblem(
        id="TRAP1_shor_factoring",
        description="Factor a 2048-bit RSA modulus into its prime components "
                    "to break the encryption.",
        expected_algorithm="none",
        expected_advantage="no_current_advantage",
        category="crypto",
        reasoning="Shor's requires millions of error-corrected qubits. Not NISQ-feasible.",
    ),
    BenchmarkProblem(
        id="TRAP2_hhl_linear",
        description="Solve a linear system of 1000 equations to find the solution vector.",
        expected_algorithm="none",
        expected_advantage="no_current_advantage",
        category="linear_algebra",
        reasoning="HHL has exponential speedup in theory but input/output bottleneck "
                  "and fault-tolerance requirements make it impractical today.",
    ),
    BenchmarkProblem(
        id="TRAP3_search_no_oracle",
        description="Find the best item in a large unsorted collection of 100000 "
                    "products based on multiple subjective criteria.",
        expected_algorithm="none",
        expected_advantage="classical_preferred",
        category="search",
        reasoning="No verifiable oracle — 'best' is subjective. Grover's doesn't apply.",
    ),

    # === EDGE CASES ===

    BenchmarkProblem(
        id="EDGE1_scheduling",
        description="Schedule 12 jobs on 3 machines to minimize the total "
                    "completion time.",
        expected_algorithm="qaoa",
        expected_advantage="potential_advantage",
        category="optimization",
        reasoning="Job scheduling is combinatorial optimization — QAOA candidate.",
    ),
    BenchmarkProblem(
        id="EDGE2_graph_coloring",
        description="Color a graph with 10 vertices using the minimum number of colors "
                    "such that no two adjacent vertices share a color.",
        expected_algorithm="qaoa",
        expected_advantage="potential_advantage",
        category="optimization",
        reasoning="Graph coloring is NP-hard combinatorial optimization.",
    ),
]
