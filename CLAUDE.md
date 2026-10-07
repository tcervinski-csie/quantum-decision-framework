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

If the new venv has no pip (`No module named pip`), the system is missing
`python3-venv`/`ensurepip`. Install it (`sudo apt install python3-venv`) or
bootstrap pip into the venv without root — see "Resuming on a New Machine".

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

If `pip` is missing inside the fresh venv, the system lacks
`python3-venv`/`ensurepip`. Either `sudo apt install python3-venv`, or bootstrap
pip into the venv without root:

```bash
curl -sS -o get-pip.py https://bootstrap.pypa.io/get-pip.py
./venv/bin/python get-pip.py
./venv/bin/python -m pip install -e ".[dev]"
```

### 2. Agent Zero setup

Agent Zero is cloned separately and run via Docker. Grant docker access first:

```bash
sudo usermod -aG docker $USER   # then log out and back in
```

A process started *before* that change keeps the old credentials and still gets
"permission denied" on the socket. Either start a fresh shell, or prefix commands
with `sg docker -c "..."`.

```bash
git clone https://github.com/frdel/agent-zero /path/to/agent-zero
cd /path/to/agent-zero
```

**The published image is not the current GitHub source.** Upstream restructured at
v0.9.9.0 (`python/helpers/` -> `helpers/`, plus a plugin architecture).
`frdel/agent-zero-run:latest` still ships the pre-v0.9.9 layout, which is what this
integration targets, so a fresh clone (v2.12+) will not match the tool code. That is
fine: the container runs its own `/a0`, and the clone only supplies the compose file
and the `usr/` mount. Do not port the integration to the cloned layout.

Create `docker-compose.yml`:

```yaml
services:
  agent-zero:
    # Derived image: base agent-zero + Qiskit (see docker/Dockerfile.agent-zero)
    build:
      context: /path/to/quantum-decision-framework
      dockerfile: docker/Dockerfile.agent-zero
    image: agent-zero-quantum:latest
    container_name: agent-zero
    ports:
      - "50080:80"
    environment:
      PYTHONPATH: /a0/quantum_ai_src
    volumes:
      - ./usr:/a0/usr
      - /path/to/quantum-decision-framework/src:/a0/quantum_ai_src
      # this image loads tools from python/tools only - there is no usr/ overlay
      - /path/to/quantum-decision-framework/agent_zero/tools/quantum_decision.py:/a0/python/tools/quantum_decision.py:ro
      - /path/to/quantum-decision-framework/agent_zero/prompts/agent.system.tool.quantum_decision.md:/a0/prompts/default/agent.system.tool.quantum_decision.md:ro
      - /path/to/quantum-decision-framework/agent_zero/extensions/_30_quantum_system.py:/a0/python/extensions/system_prompt/_30_quantum_system.py:ro
```

Then build and start:

```bash
docker compose up -d --build
```

**Do not install Qiskit with `docker exec ... uv pip install --system`.** Three
reasons, all of which fail silently:

- Agent Zero runs from the virtualenv at `/opt/venv` (Python 3.12), not the system
  interpreter (`/usr/bin/python3`, 3.13). `--system` installs into the wrong one, so
  `docker exec agent-zero python3 -c "import qiskit"` passes while the agent itself
  still cannot import it.
- An unconstrained install upgrades Agent Zero's own pins and breaks it: numpy
  1.26 -> 2.5 kills whisper/numba ("Numba needs NumPy 2.2 or less"), and pydantic
  2.10 -> 2.13 kills `mcp_server` ("cannot specify both default and
  default_factory"), which crash-loops the web UI on port 80.
- Anything installed at exec time is lost on the next `docker compose up -d`.

`docker/Dockerfile.agent-zero` handles all three: it targets `/opt/venv`, installs
against the base image's frozen versions, and bakes the result into the image.

The paths in the compose file matter: this image has no `usr/tools/` or
`usr/prompts/` convention (`agent.py`'s `get_tool` reads `python/tools` only), so
`agent_zero/install.sh`, which copies into those directories, does not apply to this
deployment and will appear to succeed while doing nothing.

Open `http://localhost:50080`, set your LLM API key in Settings.

### 3. Verify end-to-end

First confirm Agent Zero's own loaders discover the tool and the system-prompt
extension, using the interpreter the agent actually runs:

```bash
docker exec -w /a0 agent-zero /opt/venv/bin/python -c "
from python.helpers import extract_tools
from python.helpers.tool import Tool
from python.helpers.extension import Extension
print('TOOL:', [c.__name__ for c in extract_tools.load_classes_from_folder('python/tools','quantum_decision.py',Tool)])
print('EXT: ', [c.__name__ for c in extract_tools.load_classes_from_folder('python/extensions/system_prompt','*',Extension)])
"
```

Expected: `TOOL: ['QuantumDecision']`, and `QuantumSystemPrompt` listed in `EXT:`.
An empty `TOOL:` list means the mount path is wrong; a missing `QuantumSystemPrompt`
means the extension is not an `Extension` subclass (a module-level `execute()` is
silently ignored by the loader).

Then run the pipeline itself:

```bash
docker exec agent-zero /opt/venv/bin/python -c "
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
