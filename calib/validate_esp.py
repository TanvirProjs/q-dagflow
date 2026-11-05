"""Validate the ESP-based global-depolarizing bias model against density-matrix noisy simulation
with full device noise models (thermal relaxation + depolarizing + readout) from IBM snapshots."""
import json, time, numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeTorino, FakeFez
from circuits import kernel_circuit
from calib_esp import esp  # noqa

rows = []
for bname, B in [('Eagle-S', FakeSherbrooke), ('Heron-T', FakeTorino), ('Heron-F', FakeFez)]:
    b = B(); sim = AerSimulator.from_backend(b, method='density_matrix')
    for n in [3, 4, 5, 6, 7, 8, 9, 10]:
        for s in range(6):
            rng = np.random.default_rng(77 * n + s)
            x = rng.uniform(0, np.pi, n)
            # draw y near x for a spread of kernel values
            y = x + rng.normal(0, [0.05, 0.15, 0.3, 0.5, 1.0, 2.0][s], n)
            qc = kernel_circuit(n, x, y)
            k = abs(Statevector(qc.remove_final_measurements(inplace=False)).data[0]) ** 2
            tq = transpile(qc, b, optimization_level=3, seed_transpiler=s)
            F, n2q = esp(tq, b.target)
            t0 = time.time()
            res = sim.run(tq, shots=200000, seed_simulator=s).result().get_counts()
            p0 = res.get('0' * n, 0) / 200000
            rows.append(dict(backend=bname, n=n, k=float(k), F=F, n2q=n2q, p0=p0,
                             pred=F * k + (1 - F) / 2 ** n, khat=(p0 - (1 - F) / 2 ** n) / F))
            print(bname, n, s, 'k=%.3f F=%.3f p0=%.3f pred=%.3f khat=%.3f (%.1fs)' % (k, F, p0, rows[-1]['pred'], rows[-1]['khat'], time.time() - t0), flush=True)
json.dump(rows, open('../data/esp_validation.json', 'w'), indent=1)
r = np.array([[d['p0'], d['pred'], d['k'], d['khat'], d['F']] for d in rows])
ss = ((r[:, 0] - r[:, 1]) ** 2).sum(); st = ((r[:, 0] - r[:, 0].mean()) ** 2).sum()
print('R2 pred vs noisy p0 = %.4f  MAE=%.4f' % (1 - ss / st, np.abs(r[:, 0] - r[:, 1]).mean()))
print('raw |p0-k| mean=%.4f ; mitigated |khat-k| mean=%.4f' % (np.abs(r[:, 0] - r[:, 2]).mean(), np.abs(r[:, 3] - r[:, 2]).mean()))
xi = np.abs(r[:, 3] - r[:, 2]) / np.maximum(1e-9, (1 - r[:, 4]))
print('residual mismatch xi = |khat-k|/(1-F): median %.3f, p95 %.3f, max %.3f' % (np.median(xi), np.percentile(xi, 95), xi.max()))
