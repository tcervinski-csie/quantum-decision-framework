# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantum-AI is a research project implementing a **Paradigm-Aware Code Agent** — an Agent Zero integration that decides when quantum computing is appropriate for a given problem, generates Qiskit circuit code, and executes it on simulators (Qiskit Aer) or IBM Quantum hardware.

The core contribution is a **Quantum Decision Framework** with three stages:
1. **Feature Extraction** — analyze problem structure (search space, type, oracle availability)
2. **Quantum Advantage Classification** — classify as clear advantage / potential advantage / no current advantage / classical preferred
3. **Hardware Feasibility Gate** — check qubit count, circuit depth, and connectivity against available backends

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_decision_engine.py

# Run a specific test
pytest tests/test_decision_engine.py::test_grover_classification -v

# Run with coverage
pytest --cov=src/quantum_agent
```

## Architecture

```
src/quantum_agent/
├── decision_engine.py  — Problem classification and feasibility gating (core contribution)
├── code_generator.py   — Generates Qiskit circuits for selected algorithms (Grover's, QAOA)
├── executor.py         — Runs circuits on Aer simulator or IBM hardware via qiskit-ibm-runtime
└── agent_tool.py       — Agent Zero tool wrapper: ties decision → generation → execution
```

**Data flow:** User problem → `agent_tool` → `decision_engine` (classify + gate) → `code_generator` (build circuit) → `executor` (run + return results)

## Supported Quantum Algorithms

- **Grover's** — unstructured search problems with verifiable oracle
- **QAOA** — combinatorial optimization (MaxCut, portfolio optimization)
- **VQE** — molecular/quantum system simulation

## Key Constraints

- Target hardware: IBM Eagle (127 qubits). Feasibility gate must reject circuits exceeding backend limits.
- NISQ-era only: do not suggest algorithms requiring fault-tolerant QC (Shor's, HHL) for real hardware execution.
- Docker container provides the Qiskit execution environment (`docker/Dockerfile`).
