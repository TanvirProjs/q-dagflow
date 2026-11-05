import json, sys, os
sys.path.insert(0, os.getcwd())
from run_one import run
stored = json.load(open('data/exp_main.json'))
idx = {(r['family'], r['scale'], r['seed'], r['method']): r for r in stored}
cases = [('VQA-Loop','M',3), ('SQD-Hybrid','S',7), ('Epigenomics-H','M',12), ('QK-HPO','S',5), ('1000Genome-H','M',21)]
bad = 0
for f, sc, sd in cases:
    for m, kw in [('HEFT-Q', {'cams': True}), ('QG-Dyn', {'cams': True}), ('Q-DAGFlow', {'budget_factor': None}), ('LB', {})]:
        r = run(f, sc, sd, m, **kw); s = idx[(f, sc, sd, m)]
        ok = abs(r['makespan'] - s['makespan']) <= 1e-6 * max(1, s['makespan'])
        bad += not ok
        print('%-14s %s %2d %-10s rerun=%12.3f stored=%12.3f %s' % (f, sc, sd, m, r['makespan'], s['makespan'], 'OK' if ok else 'DIFF'))
print('mismatches:', bad)
