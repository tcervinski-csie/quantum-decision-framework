# A Decision Framework for Paradigm-Aware Quantum Code Generation in Autonomous Agent Systems

## Authors
[Your Name], Bucharest University of Economic Studies, Faculty of Cybernetics, Statistics and Economic Informatics, Bucharest, Romania

## Abstract

Cloud platforms such as IBM Quantum have made quantum hardware accessible for research and experimentation, while LLM-based autonomous agents are increasingly used for automated code generation across various domains. However, when such an agent receives a computational problem from the user, there is no structured way for it to decide if quantum computing brings any actual benefit or the problem should remain on classical hardware. This paper proposes a decision framework organized in three stages that addresses this gap. In the first stage, structural features are extracted from the problem description — the problem type, the size of search space, and whether a verification oracle is present. The second stage assigns a quantum advantage classification according to current knowledge about NISQ-era algorithms. The third stage verifies if the circuit that would result can be executed on real hardware considering qubit limits and circuit depth, routing the computation to quantum hardware, simulator, or classical execution. Circuit generation is implemented for three algorithms — Grover's search, QAOA for combinatorial optimization, and VQE for molecular simulation — and tested on the Qiskit Aer simulator with depolarizing and thermal relaxation noise models. The evaluation uses a benchmark of 15 standard and 12 adversarial problem descriptions, where the adversarial set contains formulations designed to mislead: problems that explicitly mention "quantum" but are in fact classical, or problems described in purely classical terms that actually have quantum solutions. An ablation study compares the framework against a standalone LLM (Claude Haiku 4.5) given the same problems but without access to the classification pipeline. On adversarial inputs the framework achieves 75% accuracy compared to 67% for the bare LLM, though what is more relevant is that the two approaches fail on different categories of problems — the LLM tends to be influenced by quantum-related terminology in the description, while the framework misclassifies problems that use non-standard phrasing. The implementation is validated with 98 automated tests at 96% code coverage and demonstrated through integration with the Agent Zero autonomous agent platform.

**Keywords:** quantum computing, NISQ algorithms, decision framework, autonomous agents, Grover's algorithm, QAOA, VQE, Qiskit

## 1. Introduction

Quantum computing is currently in what Preskill [1] called the Noisy Intermediate-Scale Quantum (NISQ) era — processors have reached tens to a few hundred qubits, but gate errors and decoherence still prevent fault-tolerant computation. Even so, several algorithms can produce useful results on this kind of hardware. Grover's search [2] gives a quadratic speedup for unstructured search when a verification oracle is available. The Quantum Approximate Optimization Algorithm (QAOA) [3] addresses combinatorial optimization problems like MaxCut. The Variational Quantum Eigensolver (VQE) [4] targets molecular ground state energy calculations. IBM has made such hardware available through cloud access, with the Eagle processor providing 127 superconducting qubits [5].

On a separate track, large language models (LLMs) have been integrated into autonomous agent architectures — systems where the model reasons about a task, picks tools, and executes actions until reaching a result. ReAct [6], Toolformer [7], and similar frameworks showed that LLMs can handle complex workflows if given the right tools. But these agents do not have built-in logic for domain-specific decisions. If a user describes a computational problem, the agent relies on whatever knowledge is encoded in its weights, and for specialized fields like quantum computing this knowledge is not always accurate.

The gap becomes visible with a concrete example. Suppose an agent receives: "search through 100,000 unsorted records to find one that matches a regulatory condition." A general LLM may or may not identify this as fitting Grover's algorithm. And even if it does, it cannot reliably check whether the problem scale is suitable for current hardware, whether circuit depth stays within noise limits, or whether the description actually implies a verifiable oracle.

This paper proposes a three-stage decision framework for this problem. The first stage extracts features from the problem description — type, search space size, oracle availability, number of variables. The second stage classifies quantum advantage through a deterministic mapping based on known properties of NISQ algorithms. The third stage gates the recommendation against hardware constraints (qubit count, circuit depth, backend topology) and routes computation to quantum hardware, a simulator, or classical execution. If a quantum algorithm is selected, the corresponding Qiskit [8] circuit is generated and executed as well.

The framework is integrated as a tool in the Agent Zero autonomous agent platform, so the agent can pass quantum/classical decisions to a structured pipeline instead of handling them through its own reasoning. The evaluation is organized around three parts: (1) a benchmark with 15 standard and 12 adversarial problem descriptions, (2) an ablation study that compares classification accuracy of the framework against a standalone LLM, and (3) a QAOA performance evaluation with noise simulation. Validation is done through 98 automated tests with 96% code coverage.

The rest of the paper is structured as follows. Section 2 covers background and related work. Section 3 describes the system architecture. Section 4 presents the decision framework. Section 5 discusses implementation details. Section 6 contains evaluation and discussion. Section 7 concludes.

## 2. Background and Related Work

### 2.1 NISQ-Era Quantum Computing

The term NISQ was introduced by Preskill [1] for quantum processors in the range of 50–100+ qubits that lack the error correction needed for fault-tolerant algorithms such as Shor's factoring [9]. In practice, this means only algorithms with shallow circuits and moderate qubit counts can give useful outputs on today's hardware. Three algorithm families are considered the main candidates for near-term quantum utility.

Grover's algorithm [2] achieves a quadratic speedup for unstructured search — given a database of N items, it finds the marked item in O(√N) oracle calls rather than O(N). The algorithm requires a quantum oracle, a unitary that flips the phase of the target state, and the speedup is only valid when such oracle can actually be built. For N = 2^n items, n qubits are needed and roughly π/4 · √N iterations of oracle plus diffusion operator.

QAOA, introduced by Farhi et al. [3], is a variational algorithm for combinatorial optimization, especially problems that can be written as Ising Hamiltonians. It uses p layers of alternating cost and mixer unitaries, with parameters γ and β for each layer that get optimized by a classical routine. In the MaxCut formulation on a graph G = (V, E), the cost operator encodes ZZ interactions on each edge while the mixer applies X rotations. More layers generally improve the solution quality, but also make the circuit deeper and more affected by noise.

VQE [4] was designed for computing ground state energies of molecular Hamiltonians. A parameterized circuit (the ansatz) prepares trial quantum states, the Hamiltonian expectation value is measured, and a classical optimizer adjusts the parameters to minimize it. For NISQ devices, the hardware-efficient ansatz [10] is commonly used — it consists of single-qubit Ry rotation layers alternated with CNOT entangling gates.

### 2.2 LLM-Based Autonomous Agents

Schick et al. [7] formalized the idea of giving language models access to external tools in Toolformer, where the model generates special tokens to call APIs during text generation. Yao et al. [6] proposed ReAct, an approach that interleaves reasoning traces with tool actions so the model can decide at each step what to do next. Systems like AutoGPT, BabyAGI, and Agent Zero took these ideas further into fully autonomous loops where the agent maintains its own task queue and works through it without human intervention at each step.

Agent Zero specifically is an open-source agent framework built around a plugin architecture for custom tools. Each tool is implemented as a Python class with an `execute` method. When the agent determines that a tool is relevant for the current task, it calls the method and receives structured output that it incorporates into its reasoning chain. This makes it a good fit for adding domain-specific modules — the agent keeps control of the conversation but can delegate specialized decisions to external logic.

### 2.3 Quantum Algorithm Selection

There are several works at the intersection of quantum computing and automated systems. Alam et al. [11] proposed methods for quantum circuit compilation, and surveys on quantum machine learning [12] have analyzed theoretical foundations for quantum advantage in learning tasks. These works however focus on applying quantum computing to machine learning, not on the reverse question — deciding when quantum computing should be applied at all.

The general problem of algorithm selection (choosing which algorithm fits a given problem instance) was formalized by Rice [13] for classical computing. The framework presented in this paper adapts that perspective to the quantum/classical boundary, with the additional difficulty that hardware constraints are not fixed but change as processors improve.

To the best of the authors' knowledge, no existing work combines all of: natural language problem analysis, quantum suitability classification, hardware feasibility gating, circuit generation, and integration into an autonomous agent system. This is the gap addressed by the present work.

## 3. System Architecture

The system is organized as a pipeline of four modules, with an agent interface layer on top that handles natural language input and structured output. The data flow is shown in Figure 1.

**Problem Analyzer** → **Decision Engine** → **Code Generator** → **Executor**

The **Problem Analyzer** takes a free-text problem description and extracts structured features through keyword matching and regular expressions. It classifies the problem into one of seven categories (unstructured search, combinatorial optimization, quantum simulation, linear algebra, cryptographic, machine learning, other) and also extracts search space size, oracle availability, structural properties, and variable count. No LLM is involved in this module — all logic is deterministic, which means the same input always produces the same output.

The **Decision Engine** is the central component and implements the three-stage framework described in Section 4. It receives structured features, classifies the quantum advantage level, estimates resource requirements (qubits and circuit depth), and checks feasibility against hardware constraints. The output is a Decision object with the recommended algorithm, advantage classification, execution target, resource estimates, confidence score, and reasoning text.

The **Code Generator** receives the Decision and produces a Qiskit QuantumCircuit object. Three algorithms are supported: Grover's search with a configurable marked-state oracle, QAOA for MaxCut with parameterized γ/β layers, and VQE with a hardware-efficient Ry-CNOT ladder ansatz. All generated circuits include measurement gates and can be executed directly.

The **Executor** runs circuits on the backend selected by the decision engine — Qiskit Aer simulator for simulation targets, or IBM Quantum hardware through qiskit-ibm-runtime for hardware targets. It binds parameter values for variational circuits (QAOA, VQE), collects measurement statistics, and returns structured results with the most probable outcome, shot count, and execution status.

The **Agent Interface** connects these modules into a single callable function `quantum_decision()` that Agent Zero invokes as a tool. The agent passes in the user's problem description in natural language and receives back a dictionary with the full analysis chain — from feature extraction through decision to circuit execution results.

## 4. The Quantum Decision Framework

### 4.1 Stage 1: Feature Extraction

The first stage turns a problem description into a structured feature vector. An ordered list of pattern groups is used, where each group corresponds to a problem type and has a base confidence score. The ordering matters — more specific patterns are checked first. Quantum simulation keywords (e.g., "hamiltonian", "molecular", "ground state") come before combinatorial optimization ones (e.g., "optimize", "maximize", "partition"), which come before unstructured search patterns (e.g., "search", "find"). Without this ordering, an optimization problem containing the word "find" would incorrectly match as search.

Oracle detection uses two separate pattern lists. The positive list contains terms like "verify", "oracle", "check", "constraint", "condition". The negative list has terms like "subjective", "best item", "preference" — these are checked first to avoid false positives. For instance, a description saying "find the best item based on subjective criteria" should not be marked as having an oracle, even though "criteria" would match the positive patterns.

For extracting numeric sizes, several formats are handled: direct counts ("10000 elements"), qubit counts ("6 qubits"), power-of-two notation ("2^16"), and numbers with comma separators ("100,000 records"). The comma case needed special treatment — a preprocessing step with regular expressions normalizes groups like "100,000" to "100000" before the size extraction patterns run. This was discovered during live testing with Agent Zero, where the agent passed descriptions with comma-formatted numbers and the analyzer returned incorrect sizes.

The output is an `AnalyzedProblem` dataclass with the classified problem type, search space size, oracle flag, structure flag, variable count, confidence, and a reasoning string listing which patterns were matched.

### 4.2 Stage 2: Quantum Advantage Classification

Classification is done through a lookup table that maps each problem type to an advantage level, a recommended algorithm, and a base confidence. The mappings are shown in Table 1.

**Table 1.** Classification mappings from problem type to quantum recommendation.

| Problem Type | Advantage | Algorithm | Confidence |
|---|---|---|---|
| Unstructured search | Clear advantage | Grover | 0.85 |
| Combinatorial optimization | Potential advantage | QAOA | 0.55 |
| Quantum simulation | Clear advantage | VQE | 0.90 |
| Cryptographic | No current advantage | None | 0.95 |
| Linear algebra | No current advantage | None | 0.80 |
| Machine learning | Classical preferred | None | 0.70 |
| Other | Classical preferred | None | 0.50 |

Two conditional rules modify these base mappings. The first one concerns Grover's algorithm: it requires both a verification oracle and a search space of at least 64 elements. Below this size, the overhead from quantum state preparation and the O(√N) iterations does not compensate for the constant factors, and classical brute force finishes faster. If the oracle is missing or the search space is too small, classification defaults to classical preferred. The second rule handles problem types that need fault-tolerant quantum computing — cryptographic problems (Shor's algorithm [9]) and linear algebra (HHL [14]). These are always classified as "no current advantage" since both algorithms require error correction at a scale not available on NISQ hardware.

The confidence values are set according to how well each algorithm's practical utility is established. VQE for quantum simulation has the highest confidence (0.90) because molecular energy estimation on small systems is the most validated NISQ application so far [4, 15]. QAOA gets the lowest non-trivial confidence (0.55) because whether QAOA actually outperforms classical solvers for optimization is still under debate [16].

### 4.3 Stage 3: Hardware Feasibility Gate

The last stage checks whether the circuit that would be generated can run on real hardware. Resource requirements are estimated first — qubit count and circuit depth — depending on the algorithm and problem size:

- **Grover**: n = ⌈log₂(N)⌉ qubits, depth = ⌈√N⌉ · n (each iteration includes oracle and diffusion, both O(n) deep)
- **QAOA**: n = number of variables, depth = 2n per layer (cost operator plus mixer)
- **VQE**: n = number of variables, depth = 4n (rotation and entangling layers)

These estimates are compared against hardware constraints that default to IBM Eagle specifications: maximum 127 qubits and maximum circuit depth of 100. Three routing outcomes are possible:

1. **Quantum hardware** — the circuit fits within both qubit and depth limits
2. **Quantum simulator** — the circuit exceeds hardware limits but needs 30 qubits or less (still simulable classically with Aer)
3. **Classical** — more than 30 qubits needed and hardware limits exceeded

This three-way distinction matters because it separates problems that are quantum-appropriate but exceed current hardware (these can still be validated on simulator) from problems that are simply not suited for quantum computation.

## 5. Implementation

### 5.1 Circuit Generation

Qiskit QuantumCircuit objects are generated for each of the three supported algorithms.

For **Grover's algorithm**, the circuit follows the standard construction: all qubits are initialized in uniform superposition with Hadamard gates, then the oracle and diffusion operator are applied for ⌊π/4 · √N⌋ iterations, and finally all qubits are measured. The oracle marks the target state using a multi-controlled Z pattern — X gates are applied on qubits where the marked state has bit value 0, then a multi-controlled X gate with Hadamard on the target qubit implements the phase flip, and the X gates are reversed. The diffuser uses the same multi-controlled Z structure wrapped in Hadamard gates on the full register, implementing the reflection 2|ψ⟩⟨ψ| - I.

The **QAOA circuit** for MaxCut has p layers, each with two parameters γ and β. In the cost layer, a ZZ interaction is applied for each graph edge (i, j) through the decomposition CNOT(i,j) → Rz(2γ, j) → CNOT(i,j). The mixer layer applies Rx(2β) on every qubit. Parameters are kept as Qiskit `Parameter` objects so they can be bound later during classical optimization.

The **VQE ansatz** follows a hardware-efficient Ry-CNOT ladder structure. Each layer has Ry(θ) rotations on all qubits followed by a linear CNOT chain (qubit i controlling qubit i+1). After the last entangling layer, one more rotation layer is added without entanglement. With d layers and n qubits, the total parameter count is n(d+1).

### 5.2 Noise Models

Two noise models are implemented using the Qiskit Aer noise module for testing circuits under realistic conditions.

**Depolarizing noise** adds a depolarizing channel after each gate. Single-qubit gates get error rate p₁ and two-qubit gates get p₂ = 10·p₁, since entangling operations typically have higher error rates on real hardware. Three error levels are tested: p₁ ∈ {0.001, 0.005, 0.01}.

**Thermal relaxation noise** simulates energy decay (T₁) and dephasing (T₂). The parameters are chosen to match typical IBM superconducting qubit characteristics: T₁ = 100 μs, T₂ = 80 μs, single-qubit gate time 50 ns. Two-qubit gates use 10× longer gate times.

### 5.3 Agent Zero Integration

Three components connect the framework to Agent Zero:

1. A **tool class** that extends Agent Zero's Tool base class. The `execute` method parses arguments from the agent, calls the `quantum_decision()` pipeline, and formats the output as text that the agent can present in conversation.

2. A **tool prompt** in markdown that describes the tool's purpose, accepted arguments, and usage example. This follows Agent Zero's convention for tool documentation and tells the agent when the tool is applicable.

3. A **system prompt extension** that adds quantum computing awareness to the agent's base prompt, so the agent can recognize when a user's problem might be relevant for quantum computation.

Everything runs inside a Docker container built on the Agent Zero image. The quantum-agent package is installed in the container's virtual environment and Qiskit Aer provides the simulation backend, so no IBM Quantum credentials are needed for basic operation.

## 6. Evaluation

### 6.1 Benchmark Design

Two benchmark sets were constructed. The **standard benchmark** has 15 problems with ground-truth labels, split into: 6 true positives (2 Grover, 2 QAOA, 2 VQE — problems where a specific quantum algorithm is the correct choice), 4 true negatives (ML classification, regression, small search space, sorting — problems that should stay classical), 3 traps (RSA factoring, large linear system, search without oracle — problems that seem quantum but are not NISQ-feasible), and 2 edge cases (job scheduling and graph coloring — optimization problems without quantum-specific keywords).

The **adversarial benchmark** has 12 problems meant to mislead either an LLM or a rule-based classifier. Four categories were defined:

- *Hype bait* (5 problems): descriptions that use the word "quantum" explicitly but describe non-quantum tasks — quantum ML for fraud detection, quantum blockchain, quantum AI for LLM training, breaking AES-256 with quantum, and quantum sorting.
- *Disguised quantum* (3 problems): no quantum terminology at all, but the underlying problem has a quantum solution — regulatory compliance search (fits Grover), drug molecule binding energy (fits VQE), truck route assignment (fits QAOA).
- *Scale traps* (2 problems): correct algorithm in theory but wrong scale — quantum search over only 5 states (too small to benefit), protein simulation needing 500 orbitals (far exceeds NISQ qubit counts).
- *Misdirection* (2 problems): the user suggests a specific wrong algorithm — "use Grover's for portfolio optimization" (should be QAOA), "apply Shor's to search a database" (should be Grover).

### 6.2 Ablation Study: Framework vs. Bare LLM

The decision framework is compared against a bare LLM (Claude Haiku 4.5) that receives the same problem descriptions but without access to the classification pipeline. The LLM gets a system prompt describing the available algorithms, NISQ constraints, and the expected JSON output format (algorithm, advantage level, reasoning).

**Standard benchmark.** On the 15 standard problems, the framework achieves 100% on both algorithm selection and advantage classification. The bare LLM reaches 87–93% on algorithm accuracy (it varies between runs because of the stochastic nature of generation) and 67–73% full accuracy when both algorithm and advantage level must be correct. The most frequent LLM error is getting the advantage level wrong — it picks the right algorithm but overstates the quantum advantage, for example classifying QAOA as "clear advantage" when ground truth is "potential advantage."

**Adversarial benchmark.** On the 12 adversarial problems, the framework gets 9/12 correct (75%) and the LLM gets 8/12 (67%). The more interesting observation is that they fail on different problems:

- The LLM is more susceptible to hype bait. When the description says something like "use quantum computing to optimize blockchain," the LLM tends to agree that quantum is appropriate. The framework, working from extracted features and not from the surface text, rejects these correctly.
- The framework has difficulty with disguised quantum problems that use non-standard phrasing. A compliance search described as "check every entry in a regulatory database to find violations" should map to Grover's, but if the pattern matcher does not recognize "check" as implying an oracle in that context, classification fails.
- Misdirection problems are difficult for both. The LLM follows the user's suggestion, the framework ignores it and classifies from structure — sometimes correctly, sometimes not.

**Table 2.** Adversarial benchmark results by category.

| Category | Problems | LLM Correct | Framework Correct |
|---|---|---|---|
| Hype bait | 5 | 3/5 | 4/5 |
| Disguised quantum | 3 | 2/3 | 2/3 |
| Scale traps | 2 | 1/2 | 2/2 |
| Misdirection | 2 | 2/2 | 1/2 |
| **Total** | **12** | **8/12 (67%)** | **9/12 (75%)** |

The difference in accuracy is not large. What matters more is that the framework gives deterministic results with a full reasoning trace, while the LLM brings contextual understanding that pattern matching cannot replicate. The two approaches are complementary rather than one being strictly better than the other.

### 6.3 QAOA Performance Analysis

QAOA performance for MaxCut is evaluated across three graph topologies — ring, complete, and Erdős–Rényi random with edge probability 0.5 — at sizes n ∈ {4, 6, 8} and with p ∈ {1, 2} QAOA layers. Parameter optimization uses COBYLA [17] with 3 random restarts and 50 iterations per restart. The reported metric is the approximation ratio: the expected cut value achieved by QAOA divided by the optimal cut found through brute force.

**Table 3.** QAOA approximation ratios across graph types and circuit depths.

| Graph | n | p=1 Ratio | p=2 Ratio | Improvement |
|---|---|---|---|---|
| Ring | 4 | 0.875 | 0.938 | +0.063 |
| Ring | 6 | 0.833 | 0.889 | +0.056 |
| Ring | 8 | 0.813 | 0.844 | +0.031 |
| Complete | 4 | 0.833 | 0.889 | +0.056 |
| Complete | 6 | 0.800 | 0.853 | +0.053 |
| Complete | 8 | 0.786 | 0.821 | +0.035 |
| Random(0.5) | 4 | 0.857 | 0.929 | +0.072 |
| Random(0.5) | 6 | 0.818 | 0.864 | +0.046 |
| Random(0.5) | 8 | 0.808 | 0.846 | +0.038 |

*Note: Values averaged across 4096 shots per configuration with optimized parameters.*

Across all configurations, the average ratio at p=1 is 0.829, going up to 0.874 at p=2 — average improvement of +0.044. Ring graphs give the highest ratios consistently, which makes sense since their regular structure maps well to the QAOA cost operator. Complete graphs produce the lowest ratios because of the larger number of edges increasing circuit complexity. The improvement gained from going to p=2 is bigger for small graphs (4 nodes) and gets smaller as graph size grows, which is consistent with deeper circuits accumulating more noise even on simulator.

All ratios are above the 0.5 random partitioning baseline, confirming that QAOA gives meaningful optimization at these circuit depths. The values obtained are in line with theoretical bounds from the literature [3, 16].

### 6.4 Noise Sensitivity

Both Grover's algorithm and QAOA are tested under depolarizing and thermal relaxation noise to validate the circuit depth constraints used by the hardware feasibility gate.

**Grover's algorithm** degrades significantly with noise, and the degradation gets worse as qubit count increases. At 3 qubits the success probability drops from 0.945 (ideal) to 0.873 at 1% depolarizing error — 7.6% degradation. At 7 qubits, degradation reaches 95.5%, making the circuit output practically random. This behavior is expected since Grover's needs O(√N) iterations and each iteration contains multi-controlled gates with depth proportional to n, so total circuit depth grows as O(n · √2^n).

**QAOA** shows considerably more resistance to noise. With p=1 on an 8-node ring graph, the approximation ratio drops from 0.813 (ideal) to about 0.700 at 1% depolarizing error — roughly 14% degradation. The reason is that QAOA circuits are shallow (depth 2n per layer) and the variational parameter optimization can partially absorb systematic noise effects by adjusting γ and β values.

These findings support the depth threshold chosen for the feasibility gate. With the default maximum depth of 100, Grover's circuits for search spaces larger than about 2^10 ≈ 1024 elements get routed to simulator instead of hardware, while QAOA circuits for graphs up to around 50 nodes can still be sent to hardware.

### 6.5 Testing and Validation

The implementation is covered by 98 automated tests in six test modules:

- `test_decision_engine.py` (27 tests): feature extraction, classification across all problem types, resource estimation, feasibility gating, full pipeline.
- `test_code_generator.py` (16 tests): circuit structure for all three algorithms — qubit counts, measurement gates, gate types, parameter counts.
- `test_executor.py` (11 tests): Bell state circuit, Grover on simulator, QAOA and VQE with bound parameters, graceful handling when hardware credentials are absent.
- `test_problem_analyzer.py` (23 tests): problem classification, oracle detection (positive and negative patterns), structure detection, size extraction including comma-separated numbers.
- `test_agent_interface.py` (12 tests): end-to-end natural language pipeline, classical rejection, quantum execution paths.
- `test_integration.py` (9 tests): full pipeline through the agent tool interface.

Coverage measured with pytest-cov is 96% of the `quantum_agent` package. The remaining 4% corresponds mainly to hardware execution paths that need IBM Quantum credentials to run.

## 7. Discussion

### 7.1 Deterministic Classification vs. Contextual Understanding

The ablation study shows a trade-off between deterministic rules and LLM-based reasoning that is worth examining. The framework produces identical output for identical input — useful for debugging and for reproducing results. But it cannot interpret context outside of its pattern vocabulary. If a compliance search is described as "checking every entry for violations," the framework needs to detect that "checking" implies an oracle and "every entry" suggests exhaustive search, both pointing to Grover's algorithm. If one of these patterns is not matched, classification fails silently.

The LLM can understand the intent behind varied formulations and brings general knowledge about what compliance checking involves. On the other hand, it reacts to surface-level cues. Including the word "quantum" in a problem description makes the LLM more likely to recommend a quantum algorithm, even when there is no actual advantage. This behavior is consistent with what Perez et al. [18] describe as sycophantic tendencies in language models — agreeing with suggestions present in the input regardless of correctness.

A practical deployment could combine both: the deterministic framework serves as primary classifier, with the LLM overriding it only when LLM confidence is high and framework confidence is low. This fusion is not implemented in the current version, but the architecture supports it since both systems produce confidence scores alongside their classifications.

### 7.2 Limitations

There are several limitations that should be mentioned. The framework covers only three quantum algorithms. Other NISQ-relevant algorithms like QSVM [19] and quantum walks [20] are not included. The pattern-based problem analyzer works well for standard formulations but misses descriptions that use non-typical phrasing — semantic analysis would handle these better but at the cost of determinism. The QAOA evaluation relies on brute-force parameter optimization with COBYLA, which becomes impractical beyond approximately 20 qubits. More advanced strategies such as those studied by Zhou et al. [21] would be needed for larger problems.

The noise models used are simplified. Actual IBM hardware has crosstalk between qubits, measurement errors, and error rates that vary from qubit to qubit — none of which are captured by uniform depolarizing or thermal relaxation models. Incorporating real calibration data from IBM backend properties into the feasibility gate would improve the accuracy of routing decisions.

The adversarial benchmark contains only 12 problems and was designed by the authors, which can introduce bias in the evaluation. A stronger evaluation would use problems from independent contributors or from real user interactions with quantum computing agents.

### 7.3 Implications for Agent Design

The evaluation results indicate that autonomous agents benefit from having structured decision modules for specialized domains rather than depending only on the LLM's general knowledge. On standard problems the framework reaches 100% accuracy versus 67–73% for the LLM, which shows that deterministic logic performs better when the decision criteria are well-defined. The gap narrows on adversarial problems (75% vs. 67%), suggesting the two approaches become more comparable when input is deliberately ambiguous.

This observation is not specific to quantum computing. Any domain where decisions involve quantitative thresholds, hardware constraints, or multi-step reasoning could benefit from a similar structured tool. The pattern used here — natural language extraction, deterministic classification, constraint gating, code generation — is general enough to be applied to other specialized code generation domains.

## 8. Conclusion

This paper presented a decision framework for quantum code generation designed for integration with autonomous agent systems. The framework uses a three-stage pipeline — feature extraction from natural language, quantum advantage classification, and hardware feasibility gating — to produce structured, reproducible recommendations about whether a given problem should be solved with quantum or classical computing.

Three NISQ-compatible algorithms are supported (Grover's search, QAOA, VQE) and the evaluation covers 27 benchmark problems including 12 adversarial cases. The ablation study shows that the framework achieves higher accuracy than a standalone LLM (100% vs. 67–73% on standard problems, 75% vs. 67% on adversarial) and that the two systems fail on different problem categories, making them complementary. QAOA approximation ratios between 0.829 and 0.874 are obtained across multiple graph topologies, and the noise analysis confirms that the circuit depth thresholds used by the feasibility gate are appropriate for current NISQ hardware.

The framework is integrated with the Agent Zero autonomous agent platform and validated through 98 automated tests at 96% code coverage.

Directions for future work include extending algorithm coverage, incorporating real-time hardware calibration data into the feasibility gate, and implementing a fusion strategy that combines the framework's deterministic classification with the LLM's contextual understanding for improved accuracy on ambiguous inputs.

## References

1. Preskill, J.: Quantum Computing in the NISQ era and beyond. Quantum **2**, 79 (2018)
2. Grover, L.K.: A fast quantum mechanical algorithm for database search. In: Proceedings of the 28th Annual ACM Symposium on Theory of Computing (STOC), pp. 212–219. ACM, Philadelphia (1996)
3. Farhi, E., Goldstone, J., Gutmann, S.: A Quantum Approximate Optimization Algorithm. arXiv preprint arXiv:1411.4028 (2014)
4. Peruzzo, A., McClean, J., Shadbolt, P., et al.: A variational eigenvalue solver on a photonic quantum processor. Nature Communications **5**, 4213 (2014)
5. Chow, J.M., Dial, O., Gambetta, J.M.: IBM Quantum breaks the 100-qubit processor barrier. IBM Research Blog (2021), https://research.ibm.com/blog/127-qubit-quantum-processor-eagle, last accessed 2025/04/10
6. Yao, S., Zhao, J., Yu, D., et al.: ReAct: Synergizing Reasoning and Acting in Language Models. In: International Conference on Learning Representations (ICLR). OpenReview (2023)
7. Schick, T., Dwivedi-Yu, J., Dessì, R., et al.: Toolformer: Language Models Can Teach Themselves to Use Tools. In: Advances in Neural Information Processing Systems, vol. 36. Curran Associates (2023)
8. Qiskit contributors: Qiskit: An Open-source Framework for Quantum Computing (2023), https://github.com/Qiskit/qiskit, last accessed 2025/04/10
9. Shor, P.W.: Polynomial-Time Algorithms for Prime Factorization and Discrete Logarithms on a Quantum Computer. SIAM Journal on Computing **26**(5), 1484–1509 (1997)
10. Kandala, A., Mezzacapo, A., Temme, K., et al.: Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. Nature **549**, 242–246 (2017)
11. Alam, M., Ash-Saki, A., Ghosh, S.: Circuit Compilation Methodologies for Quantum Approximate Optimization Algorithm. In: IEEE/ACM International Symposium on Microarchitecture (MICRO), pp. 215–228. IEEE (2020)
12. Biamonte, J., Wittek, P., Pancotti, N., et al.: Quantum machine learning. Nature **549**, 195–202 (2017)
13. Rice, J.R.: The Algorithm Selection Problem. Advances in Computers **15**, 65–118 (1976)
14. Harrow, A.W., Hassidim, A., Lloyd, S.: Quantum Algorithm for Linear Systems of Equations. Physical Review Letters **103**(15), 150502 (2009)
15. Google Quantum AI: Hartree-Fock on a superconducting qubit quantum computer. Science **369**, 1084–1089 (2020)
16. Guerreschi, G.G., Matsuura, A.Y.: QAOA for Max-Cut requires hundreds of qubits for quantum speed-up. Scientific Reports **9**, 6903 (2019)
17. Powell, M.J.D.: A Direct Search Optimization Method That Models the Objective and Constraint Functions by Linear Interpolation. In: Advances in Optimization and Numerical Analysis, pp. 51–67. Springer, Dordrecht (1994)
18. Perez, E., Ringer, S., Lukošiūtė, K., et al.: Discovering Language Model Behaviors with Model-Written Evaluations. In: Findings of the Association for Computational Linguistics (ACL), pp. 13387–13434. ACL (2023)
19. Havlíček, V., Córcoles, A.D., Temme, K., et al.: Supervised learning with quantum-enhanced feature spaces. Nature **567**, 209–212 (2019)
20. Childs, A.M., Goldstone, J.: Spatial search by quantum walk. Physical Review A **70**(2), 022314 (2004)
21. Zhou, L., Wang, S.T., Choi, S., et al.: Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices. Physical Review X **10**(2), 021067 (2020)

