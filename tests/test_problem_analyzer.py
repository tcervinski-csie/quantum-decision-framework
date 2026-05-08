"""Tests for the natural language problem analyzer."""

import pytest
from quantum_agent.problem_analyzer import analyze_problem, AnalyzedProblem
from quantum_agent.decision_engine import ProblemType


class TestProblemClassification:
    def test_grover_search_description(self):
        result = analyze_problem(
            "I need to search through an unsorted database of 10000 elements "
            "to find one that satisfies a specific condition."
        )
        assert result.problem_type == ProblemType.UNSTRUCTURED_SEARCH
        assert result.has_oracle is True  # "satisfies" implies verifier
        assert result.search_space_size == 10000

    def test_maxcut_optimization(self):
        result = analyze_problem(
            "Solve the MaxCut problem on a graph with 8 nodes "
            "to find the optimal partition."
        )
        assert result.problem_type == ProblemType.COMBINATORIAL_OPTIMIZATION
        assert result.num_variables == 8

    def test_tsp_optimization(self):
        result = analyze_problem(
            "Find the shortest route for a traveling salesman "
            "visiting 12 cities."
        )
        assert result.problem_type == ProblemType.COMBINATORIAL_OPTIMIZATION

    def test_molecular_simulation(self):
        result = analyze_problem(
            "Simulate the ground state energy of a hydrogen molecule "
            "using 4 qubits."
        )
        assert result.problem_type == ProblemType.QUANTUM_SIMULATION
        assert result.num_variables == 4

    def test_factoring_cryptographic(self):
        result = analyze_problem(
            "Factor a large RSA number into its prime components."
        )
        assert result.problem_type == ProblemType.CRYPTOGRAPHIC

    def test_linear_system(self):
        result = analyze_problem(
            "Solve a linear system of equations Ax = b where A is a 64x64 matrix."
        )
        assert result.problem_type == ProblemType.LINEAR_ALGEBRA

    def test_ml_classification(self):
        result = analyze_problem(
            "Train a classification model using deep learning on image data."
        )
        assert result.problem_type == ProblemType.MACHINE_LEARNING

    def test_unknown_problem(self):
        result = analyze_problem("Make me a sandwich.")
        assert result.problem_type == ProblemType.OTHER
        assert result.confidence < 0.5

    def test_portfolio_optimization(self):
        result = analyze_problem(
            "Optimize a portfolio of 20 assets to minimize risk "
            "while maximizing return."
        )
        assert result.problem_type == ProblemType.COMBINATORIAL_OPTIMIZATION


class TestOracleDetection:
    def test_explicit_oracle(self):
        result = analyze_problem("Search with an oracle that verifies solutions.")
        assert result.has_oracle is True

    def test_constraint_implies_oracle(self):
        result = analyze_problem(
            "Find an element that satisfies the constraint x^2 mod N = 1."
        )
        assert result.has_oracle is True

    def test_no_oracle_for_optimization(self):
        result = analyze_problem(
            "Optimize the cut value of a graph partition."
        )
        assert result.has_oracle is False


class TestStructureDetection:
    def test_periodic_structure(self):
        result = analyze_problem(
            "Find the period of a function f(x) over a cyclic group."
        )
        assert result.has_structure is True

    def test_no_structure(self):
        result = analyze_problem("Search an unsorted list of items.")
        assert result.has_structure is False


class TestSizeExtraction:
    def test_element_count(self):
        result = analyze_problem("Search through 5000 elements in a database.")
        assert result.search_space_size == 5000

    def test_qubit_count(self):
        result = analyze_problem("Simulate a 6 qubit quantum system.")
        assert result.num_variables == 6
        assert result.search_space_size == 64  # 2^6

    def test_node_count(self):
        result = analyze_problem("Graph with 10 nodes, find the max cut.")
        assert result.num_variables == 10

    def test_power_of_two(self):
        result = analyze_problem("Search space of size 2^16.")
        assert result.search_space_size == 65536

    def test_comma_separated_numbers(self):
        result = analyze_problem("Search through 100,000 records in a database.")
        assert result.search_space_size == 100000

    def test_large_comma_number(self):
        result = analyze_problem("Find one item among 1,000,000 elements.")
        assert result.search_space_size == 1000000

    def test_default_size_when_unspecified(self):
        result = analyze_problem("Search for an item in a database.")
        assert result.search_space_size == 1024  # default


class TestAnalyzedProblemOutput:
    def test_has_reasoning(self):
        result = analyze_problem("Search 1000 elements with a verification oracle.")
        assert len(result.reasoning) > 0

    def test_confidence_range(self):
        result = analyze_problem("Optimize a MaxCut problem on 8 nodes.")
        assert 0.0 <= result.confidence <= 1.0
