"""Example: Grover's search for a constraint satisfaction problem.

Problem: Find a 4-bit assignment (x0, x1, x2, x3) where:
  - x0 XOR x1 = 1  (exactly one of x0, x1 is 1)
  - x2 AND x3 = 1  (both x2 and x3 must be 1)
  - x0 OR x2 = 1   (at least one of x0, x2 is 1)

Solutions: |0111> (x0=0, x1=1, x2=1, x3=1) and |1011> (x0=1, x1=0, x2=1, x3=1)
           Note: bit ordering is x3 x2 x1 x0 in Qiskit (LSB)

This is a realistic use case: boolean SAT with a verifiable oracle.
The oracle marks states that satisfy ALL constraints simultaneously.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from math import pi, sqrt


def build_oracle(n: int = 4) -> QuantumCircuit:
    """Build an oracle that marks states satisfying the constraints.

    Uses an ancilla-based approach:
    - 4 data qubits (x0..x3)
    - 3 ancilla qubits (one per constraint)
    - 1 target qubit for phase kickback

    The oracle flips the phase of states where all constraints are satisfied.
    """
    # 4 data + 3 ancilla + 1 target = 8 qubits
    n_total = 8
    oracle = QuantumCircuit(n_total, name="csp_oracle")

    data = list(range(4))      # x0, x1, x2, x3
    anc = list(range(4, 7))    # constraint ancillae
    target = 7                  # phase kickback target

    # --- Compute constraints into ancillae ---

    # Constraint 1: x0 XOR x1 = 1
    # XOR = CNOT from both inputs to ancilla (initialized to |0>)
    # If XOR=1, ancilla becomes |1>
    oracle.cx(data[0], anc[0])
    oracle.cx(data[1], anc[0])

    # Constraint 2: x2 AND x3 = 1
    # AND = Toffoli gate
    oracle.ccx(data[2], data[3], anc[1])

    # Constraint 3: x0 OR x2 = 1
    # OR: flip inputs, AND, flip result and inputs
    # OR(a,b) = NOT(AND(NOT(a), NOT(b)))
    oracle.x(data[0])
    oracle.x(data[2])
    oracle.ccx(data[0], data[2], anc[2])
    oracle.x(data[0])
    oracle.x(data[2])
    oracle.x(anc[2])  # flip to get OR

    # --- Multi-controlled phase flip: all ancillae must be |1> ---
    # Use multi-controlled Z on target (prepared in |-> for phase kickback)
    oracle.mcx(anc, target)

    # --- Uncompute ancillae ---
    # Constraint 3 uncompute
    oracle.x(anc[2])
    oracle.x(data[0])
    oracle.x(data[2])
    oracle.ccx(data[0], data[2], anc[2])
    oracle.x(data[0])
    oracle.x(data[2])

    # Constraint 2 uncompute
    oracle.ccx(data[2], data[3], anc[1])

    # Constraint 1 uncompute
    oracle.cx(data[1], anc[0])
    oracle.cx(data[0], anc[0])

    return oracle


def build_diffuser(n: int = 4) -> QuantumCircuit:
    """Grover diffusion operator on the data qubits only."""
    n_total = 8  # must match oracle
    diffuser = QuantumCircuit(n_total, name="diffuser")

    data = list(range(n))

    diffuser.h(data)
    diffuser.x(data)

    # Multi-controlled Z on data qubits using ancilla
    diffuser.h(data[-1])
    diffuser.mcx(data[:-1], data[-1])
    diffuser.h(data[-1])

    diffuser.x(data)
    diffuser.h(data)

    return diffuser


def verify_solution(x0, x1, x2, x3) -> bool:
    """Check if an assignment satisfies all constraints."""
    c1 = (x0 ^ x1) == 1
    c2 = (x2 & x3) == 1
    c3 = (x0 | x2) == 1
    return c1 and c2 and c3


def main():
    print("=" * 65)
    print("Grover's Search — Boolean Constraint Satisfaction Problem")
    print("=" * 65)
    print("\nConstraints:")
    print("  C1: x0 XOR x1 = 1")
    print("  C2: x2 AND x3 = 1")
    print("  C3: x0 OR  x2 = 1")

    # Find solutions classically for verification
    print("\nClassical brute-force solutions:")
    solutions = []
    for i in range(16):
        bits = [(i >> b) & 1 for b in range(4)]
        if verify_solution(*bits):
            # Qiskit bit ordering: qubit 0 is rightmost
            bs = format(i, "04b")[::-1]
            solutions.append(bs)
            print(f"  x0={bits[0]}, x1={bits[1]}, x2={bits[2]}, x3={bits[3]} → |{bs}> (Qiskit ordering)")
    print(f"  Total: {len(solutions)} solutions out of 16")

    # Build Grover circuit
    n_data = 4
    n_total = 8
    N = 2 ** n_data  # search space
    M = len(solutions)  # number of solutions
    n_iterations = max(1, round(pi / 4 * sqrt(N / M)))

    print(f"\nGrover's circuit:")
    print(f"  Data qubits:    {n_data}")
    print(f"  Ancilla qubits: 4 (3 constraint + 1 phase)")
    print(f"  Search space:   {N}")
    print(f"  Solutions:      {M}")
    print(f"  Iterations:     {n_iterations}")

    # Build circuit
    qc = QuantumCircuit(n_total, n_data)

    # Initialize data qubits in superposition
    qc.h(range(n_data))

    # Initialize phase kickback target in |->
    qc.x(7)
    qc.h(7)

    qc.barrier()

    oracle = build_oracle()
    diffuser = build_diffuser()

    for _ in range(n_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
        qc.barrier()

    # Measure data qubits only
    qc.measure(range(n_data), range(n_data))

    print(f"  Circuit depth:  {qc.depth()}")
    print(f"  Total gates:    {len(qc.data)}")

    # Execute
    shots = 4096
    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts(qc)

    print(f"\n--- Results ({shots} shots) ---")
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    for bitstring, count in sorted_counts[:8]:
        pct = count / shots * 100
        # Check if this is a valid solution
        bits = [int(b) for b in bitstring[::-1]]  # reverse for x0..x3
        is_valid = verify_solution(*bits) if len(bits) == 4 else False
        marker = " ← SOLUTION" if is_valid else ""
        print(f"  |{bitstring}> : {count:4d} ({pct:5.1f}%){marker}")

    # Success probability: check each measured bitstring against constraints
    solution_counts = 0
    for bitstring, count in counts.items():
        # Qiskit returns bitstrings with qubit 0 on the right
        bits = [int(b) for b in reversed(bitstring)]
        if len(bits) == 4 and verify_solution(*bits):
            solution_counts += count
    success_prob = solution_counts / shots
    print(f"\n  Total solution probability: {success_prob:.1%}")
    print(f"  Classical random guess:     {M/N:.1%}")
    print(f"  Quantum speedup factor:     {success_prob / (M/N):.1f}x")


if __name__ == "__main__":
    main()
