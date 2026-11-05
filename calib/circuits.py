"""Circuit templates for quantum tasks in hybrid AI workflows."""
import numpy as np
from qiskit import QuantumCircuit

def feature_map(n, x, reps=2):
    """ZZ-feature-map style embedding U(x) with linear entanglement (Havlicek et al. 2019)."""
    qc = QuantumCircuit(n)
    for _ in range(reps):
        qc.h(range(n))
        for i in range(n):
            qc.p(2.0 * x[i], i)
        for i in range(n - 1):
            qc.cx(i, i + 1)
            qc.p(2.0 * (np.pi - x[i]) * (np.pi - x[i + 1]), i + 1)
            qc.cx(i, i + 1)
    return qc

def kernel_circuit(n, x, y, reps=2, measure=True):
    """Fidelity-kernel circuit U(y)^dagger U(x); kernel = Pr[0^n]."""
    qc = feature_map(n, x, reps).compose(feature_map(n, y, reps).inverse())
    if measure:
        qc.measure_all()
    return qc

def hea(n, theta, layers=2, measure=True):
    """Hardware-efficient ansatz (RY + linear CZ) used for VQA gradient estimation."""
    qc = QuantumCircuit(n)
    k = 0
    for _ in range(layers):
        for i in range(n):
            qc.ry(theta[k % len(theta)], i); k += 1
        for i in range(n - 1):
            qc.cz(i, i + 1)
    for i in range(n):
        qc.ry(theta[k % len(theta)], i); k += 1
    if measure:
        qc.measure_all()
    return qc

def sampler_ansatz(n, theta, layers=4, measure=True):
    """Deeper brick-wall ansatz approximating LUCJ-style sampling circuits (SQD workflows)."""
    qc = QuantumCircuit(n)
    k = 0
    for l in range(layers):
        for i in range(n):
            qc.rz(theta[k % len(theta)], i); qc.ry(theta[(k + 1) % len(theta)], i); k += 2
        start = l % 2
        for i in range(start, n - 1, 2):
            qc.cx(i, i + 1)
    if measure:
        qc.measure_all()
    return qc

TEMPLATES = {
    'kernel': lambda n, rng: kernel_circuit(n, rng.uniform(0, np.pi, n), rng.uniform(0, np.pi, n)),
    'vqa': lambda n, rng: hea(n, rng.uniform(0, 2 * np.pi, 3 * n), layers=2),
    'sqd': lambda n, rng: sampler_ansatz(n, rng.uniform(0, 2 * np.pi, 8 * n), layers=4),
}
