"""Build ESP / duration tables for circuit templates on real IBM calibration snapshots."""
import json, numpy as np
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeKyiv, FakeTorino, FakeFez
from circuits import TEMPLATES

BACKENDS = {'Eagle-S': FakeSherbrooke, 'Eagle-K': FakeKyiv, 'Heron-T': FakeTorino, 'Heron-F': FakeFez}
NS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32, 40, 48, 56, 64, 80, 96]

def esp(tq, target):
    """Estimated success probability (Nishio et al. 2020): product of (1-error) over all ops."""
    logf = 0.0; n2q = 0
    for inst in tq.data:
        name = inst.operation.name
        if name in ('barrier', 'delay', 'rz'):
            continue
        qargs = tuple(tq.find_bit(q).index for q in inst.qubits)
        props = target[name].get(qargs) if name in target.operation_names else None
        err = props.error if (props is not None and props.error is not None) else 0.0
        logf += np.log(max(1e-12, 1.0 - err))
        if len(qargs) == 2:
            n2q += 1
    return float(np.exp(logf)), n2q

if __name__ == "__main__":
    out = {}
    for bname, B in BACKENDS.items():
        b = B(); tgt = b.target
        out[bname] = {'num_qubits': b.num_qubits, 'source': b.name, 'rows': []}
        for tmpl, f in TEMPLATES.items():
            for n in (NS if tmpl != 'kernel' else [x for x in NS if x <= 40]):
                vals = []
                for s in range(3):
                    rng = np.random.default_rng(1000 * n + s)
                    tq = transpile(f(n, rng), b, optimization_level=3, seed_transpiler=s)
                    F, n2q = esp(tq, tgt)
                    vals.append((F, n2q, tq.estimate_duration(tgt), tq.depth()))
                v = np.array(vals)
                out[bname]['rows'].append({'template': tmpl, 'n': n, 'esp': float(v[:, 0].mean()), 'esp_sd': float(v[:, 0].std()),
                                           'n2q': float(v[:, 1].mean()), 'duration_s': float(v[:, 2].mean()), 'depth': float(v[:, 3].mean())})
                print(bname, tmpl, n, 'ESP=%.3f n2q=%d dur=%.1fus' % (v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean() * 1e6), flush=True)
    json.dump(out, open('../data/esp_table.json', 'w'), indent=1)
