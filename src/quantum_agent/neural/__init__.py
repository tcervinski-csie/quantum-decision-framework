"""Learned backend for Stage 2 of the decision framework.

The network replaces `classify_advantage` only. Stage 1 (feature extraction) and
Stage 3 (the hardware feasibility gate) stay deterministic — the gate encodes hard
physical limits, and an approximation of "<= 127 qubits" would eventually admit an
unrunnable circuit.

torch is an optional dependency (`pip install -e ".[neural]"`). Nothing in this
package imports it unless the neural backend is explicitly requested.
"""
