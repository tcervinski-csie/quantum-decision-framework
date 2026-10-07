"""Natural language problem analyzer.

Extracts structured problem features from free-text descriptions
so the decision engine can classify them. Uses keyword matching and
pattern recognition — no LLM dependency for the core logic.

The agent (LLM) can use this directly, or override with its own parsing
when it has higher confidence about the problem structure.
"""

import re
from dataclasses import dataclass
from typing import Optional

from quantum_agent.decision_engine import ProblemType


@dataclass
class AnalyzedProblem:
    """Result of analyzing a natural language problem description."""
    problem_type: ProblemType
    search_space_size: int
    has_oracle: bool
    has_structure: bool
    num_variables: int
    confidence: float
    reasoning: str


# Keyword patterns mapped to problem types, ordered by specificity.
_PROBLEM_PATTERNS: list[tuple[ProblemType, list[str], float]] = [
    # Quantum simulation — most specific
    (ProblemType.QUANTUM_SIMULATION, [
        r"molecul\w*", r"hamiltonian", r"eigenvalu\w*", r"ground.?state",
        r"quantum.?simulat\w*", r"chemical", r"energy.?level",
        r"spin.?chain", r"fermi\w*", r"boson\w*",
    ], 0.85),

    # Cryptographic
    (ProblemType.CRYPTOGRAPHIC, [
        r"factor\w*(?:ization)?", r"rsa", r"cryptograph\w*", r"prime",
        r"discrete.?log\w*", r"elliptic.?curve",
    ], 0.90),

    # Combinatorial optimization — check BEFORE unstructured search
    # since optimization problems often contain "find" which would match search.
    (ProblemType.COMBINATORIAL_OPTIMIZATION, [
        r"optimi[sz]\w*", r"max\w*\.?cut", r"min\w*\.?cut", r"travel\w*salesman",
        r"tsp", r"routing", r"portfolio", r"schedul\w*", r"partition\w*",
        r"knapsack", r"graph.?color\w*", r"color\w*.*graph", r"combinat\w*", r"vertex.?cover",
        r"shortest.?(?:route|path)", r"travel\w*\s+salesman",
        r"minimi[sz]\w*.*(?:cost|time|risk|distance)",
        r"maximi[sz]\w*.*(?:return|profit|cut|value|edge)",
        r"job\w*\s+(?:on|across)\s+\d+\s+machine",
        r"allocat\w*",
    ], 0.80),

    # Unstructured search
    (ProblemType.UNSTRUCTURED_SEARCH, [
        r"search\w*", r"find.?(?:an?|the|one)", r"lookup", r"unsorted",
        r"database.?search", r"needle.?in", r"brute.?force",
        r"satisf\w*(?:ability)?", r"constraint",
    ], 0.75),

    # Linear algebra
    (ProblemType.LINEAR_ALGEBRA, [
        r"linear.?system", r"matrix.?invers\w*",
        r"linear.?equation", r"solve.*Ax\s*=\s*b",
        r"system\s+of\s+\d*\s*equations",
    ], 0.75),

    # Machine learning
    (ProblemType.MACHINE_LEARNING, [
        r"machine.?learn\w*", r"classif\w*", r"regression", r"neural",
        r"deep.?learn\w*", r"train\w*model", r"svm", r"cluster\w*",
        r"predict\w*",
    ], 0.70),
]

# Patterns that suggest an oracle/verifier exists.
_ORACLE_PATTERNS = [
    r"verif\w*", r"oracle", r"check\w*(?:able)?", r"valid\w*",
    r"satisf\w*", r"constraint", r"condition",
]

# Patterns that explicitly negate oracle presence.
_NO_ORACLE_PATTERNS = [
    r"subjective", r"best\s+(?:item|option|choice)", r"preference",
    r"rank\w*", r"opinion",
]

# Patterns that suggest mathematical structure.
_STRUCTURE_PATTERNS = [
    r"period\w*", r"symmetr\w*", r"group", r"fourier",
    r"cyclic", r"regular", r"structured",
]

# Allows a short run of qualifiers between the number and the noun it counts, so
# "65,536 employee records" reads the same as "65,536 records". Bounded at two
# words and restricted to alphabetic tokens to avoid matching across clauses.
_QUALIFIERS = r"(?:[a-z][a-z-]*\s+){0,2}?"

# Patterns to extract numeric sizes.
_SIZE_PATTERNS = [
    (rf"(\d+)\s*{_QUALIFIERS}(?:qubit|spin|atom|node|vert\w*|variable|cit\w*)", "num_variables"),
    (rf"(\d+)\s*{_QUALIFIERS}(?:element|item|record|entr\w*|row)", "search_space"),
    (r"(?:space|size|domain)\s*(?:of|is|=|:)\s*(\d+)", "search_space"),
    (rf"(\d+)\s*{_QUALIFIERS}(?:bit|digit)", "bits"),
    (r"2\s*\^\s*(\d+)", "power_of_two"),
]


def analyze_problem(description: str) -> AnalyzedProblem:
    """Analyze a natural language problem description.

    Extracts problem type, estimated size, oracle availability,
    and structural properties from free text.
    """
    text = description.lower().strip()

    # Classify problem type
    problem_type, type_confidence, type_reasoning = _classify_type(text)

    # Detect oracle
    has_oracle, oracle_reasoning = _detect_oracle(text)

    # Detect structure
    has_structure = _detect_structure(text)

    # Extract sizes
    search_space, num_variables = _extract_sizes(text)

    reasoning_parts = [type_reasoning]
    if oracle_reasoning:
        reasoning_parts.append(oracle_reasoning)
    if has_structure:
        reasoning_parts.append("Structural patterns detected.")

    return AnalyzedProblem(
        problem_type=problem_type,
        search_space_size=search_space,
        has_oracle=has_oracle,
        has_structure=has_structure,
        num_variables=num_variables,
        confidence=type_confidence,
        reasoning=" ".join(reasoning_parts),
    )


def _classify_type(text: str) -> tuple[ProblemType, float, str]:
    """Match text against problem type patterns."""
    best_type = ProblemType.OTHER
    best_score = 0.0
    best_confidence = 0.5
    matched_keywords: list[str] = []

    for ptype, patterns, base_confidence in _PROBLEM_PATTERNS:
        score = 0
        hits = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                score += len(matches)
                hits.append(pattern)

        if score > best_score:
            best_score = score
            best_type = ptype
            best_confidence = min(base_confidence + 0.02 * (score - 1), 0.95)
            matched_keywords = hits

    if best_score == 0:
        return (ProblemType.OTHER, 0.3, "No recognized problem patterns found.")

    return (
        best_type,
        best_confidence,
        f"Classified as {best_type.value} (matched: {', '.join(matched_keywords[:3])}).",
    )


def _detect_oracle(text: str) -> tuple[bool, str]:
    """Check if the problem description implies a verifiable oracle."""
    # Check for explicit negation first
    for pattern in _NO_ORACLE_PATTERNS:
        if re.search(pattern, text):
            return (False, "Subjective criteria detected — no formal oracle.")

    for pattern in _ORACLE_PATTERNS:
        if re.search(pattern, text):
            return (True, f"Oracle implied by '{pattern}' pattern.")
    return (False, "")


def _detect_structure(text: str) -> bool:
    """Check if the problem has exploitable mathematical structure."""
    for pattern in _STRUCTURE_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def _parse_number(s: str) -> int:
    """Parse a number string that may contain commas or underscores."""
    return int(s.replace(",", "").replace("_", ""))


def _extract_sizes(text: str) -> tuple[int, int]:
    """Extract search space size and number of variables from text.

    Returns (search_space_size, num_variables).
    """
    # Normalize comma-separated numbers: "100,000" → "100000"
    normalized = re.sub(r"(\d{1,3}(?:,\d{3})+)", lambda m: m.group().replace(",", ""), text)

    search_space = 1024  # default
    num_variables = 0

    for pattern, kind in _SIZE_PATTERNS:
        match = re.search(pattern, normalized)
        if match:
            value = int(match.group(1))
            if kind == "num_variables":
                num_variables = value
                if search_space == 1024:  # not yet set
                    search_space = 2 ** value
            elif kind == "search_space":
                search_space = value
            elif kind == "bits":
                num_variables = value
                search_space = 2 ** value
            elif kind == "power_of_two":
                search_space = 2 ** value
                num_variables = value

    return (search_space, num_variables)
