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

## Current State (as of May 2026)

- Core framework complete: 98 tests, 96% coverage
- Benchmarks complete: 15/15 standard accuracy, adversarial benchmark showing complementary failure modes vs bare LLM
- Paper published at CONFIE 2026
- Agent Zero integration complete (tool file, prompts, system prompt extension)

### What's left to do

1. **Extend algorithm coverage** — QSVM and quantum walks are the natural next additions
2. **Real IBM hardware validation** — executor path is implemented but untested without credentials
3. **Hybrid fusion strategy** — combine deterministic framework (primary) with LLM override when LLM confidence is high and framework confidence is low; both already produce confidence scores so architecture supports it
4. **Real-time hardware calibration** — poll IBM Quantum API for live backend properties instead of hardcoded Eagle limits
5. **Larger adversarial benchmark** — current 12-problem set was author-constructed; independent curation would strengthen the evaluation

## Resuming on a New Machine

### 1. Clone and install

```bash
git clone https://github.com/tcervinski-csie/quantum-decision-framework
cd quantum-decision-framework
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest  # should be 98 tests passing
```

### 2. Agent Zero setup

Agent Zero must be cloned separately and run via Docker:

```bash
git clone https://github.com/frdel/agent-zero /path/to/agent-zero
cd /path/to/agent-zero
```

Create `docker-compose.yml`:

```yaml
services:
  agent-zero:
    image: frdel/agent-zero-run:latest
    container_name: agent-zero
    ports:
      - "50080:80"
    volumes:
      - ./usr:/a0/usr
      - /path/to/quantum-decision-framework/src:/a0/quantum_ai_src
```

Then start and install Qiskit inside the container:

```bash
docker compose up -d
docker exec agent-zero uv pip install --system --break-system-packages qiskit qiskit-aer qiskit-ibm-runtime
docker exec agent-zero uv pip install --system --break-system-packages -e /a0/quantum_ai_src  # if pyproject.toml is mounted
```

Copy tool files into Agent Zero:

```bash
cp agent_zero/tools/quantum_decision.py /path/to/agent-zero/usr/tools/
cp agent_zero/prompts/agent.system.tool.quantum_decision.md /path/to/agent-zero/usr/prompts/
```

Or run the installer:

```bash
bash agent_zero/install.sh /path/to/agent-zero
```

Open `http://localhost:50080`, set your LLM API key in Settings.

### 3. Verify end-to-end

```bash
# Verify quantum tool works inside container
docker exec agent-zero python3 -c "
import sys; sys.path.insert(0, '/a0/quantum_ai_src')
from quantum_agent.agent_interface import quantum_decision
r = quantum_decision('Search 1024 records for one matching a condition')
print(r['decision']['algorithm'], r['decision']['advantage'])
"
```

Expected output: `grover clear_advantage`

### 4. Demo prompts for Agent Zero

**Quantum (should trigger Grover's):**
> "I have an unsorted database of 65,536 employee records and need to find the one where the employee ID matches a compliance flag. Each record can be verified in constant time. Should I use quantum computing, and can you run it?"

**Classical (should reject quantum):**
> "I need to classify 50,000 customer records into churn vs. no-churn using their purchase history. Which algorithm should I use?"
