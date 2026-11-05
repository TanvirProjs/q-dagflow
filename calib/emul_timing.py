"""Measure classical statevector-emulation cost on the measurement host (genuine timings)."""
import json, time, numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from circuits import feature_map, hea, sampler_ansatz

sim = AerSimulator(method='statevector', max_parallel_threads=2, precision='double')
rows = []
for tmpl in ['kernel', 'vqa', 'sqd']:
    for n in [8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 26]:
        rng = np.random.default_rng(n)
        if tmpl == 'kernel':
            qc = feature_map(n, rng.uniform(0, np.pi, n))           # one embedding statevector U(x)|0>
        elif tmpl == 'vqa':
            qc = hea(n, rng.uniform(0, 2 * np.pi, 3 * n), measure=False)
        else:
            qc = sampler_ansatz(n, rng.uniform(0, 2 * np.pi, 8 * n), measure=False)
        qc.save_statevector()
        tq = transpile(qc, sim, optimization_level=0)
        reps = 5 if n <= 20 else 2
        ts = []
        for r in range(reps):
            t0 = time.perf_counter(); res = sim.run(tq).result(); ts.append(time.perf_counter() - t0)
            sv = np.asarray(res.get_statevector())
        t_sv = float(np.median(ts))
        t0 = time.perf_counter(); [np.vdot(sv, sv) for _ in range(3)]; t_ip = (time.perf_counter() - t0) / 3
        rows.append(dict(template=tmpl, n=n, t_sv=t_sv, t_ip=t_ip, gates=sum(qc.count_ops().values())))
        print(tmpl, n, 't_sv=%.4fs t_ip=%.5fs' % (t_sv, t_ip), flush=True)
        del sv, res
json.dump(rows, open('../data/emulation_timing.json', 'w'), indent=1)
