"""Experiment drivers. Usage: python experiments.py <name>"""
import sys, json
from sweep import sweep
FAMS = ['QK-HPO', 'VQA-Loop', 'SQD-Hybrid', 'Montage-H', 'Epigenomics-H', '1000Genome-H']
BASE = ['HEFT-Q', 'PEFT-Q', 'SHEFT-Q', 'QG-Dyn', 'FG-Dyn', 'CF-Dyn']
C = {'cams': True}
QD = {'budget_factor': None}

def main():
    tasks = []
    for sc in ['S', 'M', 'L']:
        for f in FAMS:
            for s in range(1, 31):
                tasks.append((f, sc, s, 'LB', 'LB', {}))
                for b in BASE:
                    tasks.append((f, sc, s, b, b, C))
                tasks.append((f, sc, s, 'Q-DAGFlow', 'Q-DAGFlow', QD))
    sweep(tasks, 'data/exp_main.json')

def practice():
    """Precision reliability: baselines as practised (published calibration) vs CAMS; drift stress."""
    tasks = []
    for f in FAMS:
        for s in range(1, 31):
            for b in ['HEFT-Q', 'PEFT-Q', 'QG-Dyn']:
                tasks.append((f, 'M', s, b + '[pub]', b, {}))
    for ds in [0.0, 0.5, 1.0, 2.0, 4.0]:
        for f in ['VQA-Loop', 'SQD-Hybrid', 'Montage-H']:
            for s in range(1, 16):
                ek = {'env_kw': {'drift_scale': ds}}
                tasks.append((f, 'M', s, 'QG-Dyn[pub]|%g' % ds, 'QG-Dyn', dict(ek)))
                tasks.append((f, 'M', s, 'QG-Dyn[CAMS]|%g' % ds, 'QG-Dyn', dict(ek, cams=True)))
                tasks.append((f, 'M', s, 'Q-DAGFlow|%g' % ds, 'Q-DAGFlow', dict(ek, budget_factor=None)))
                tasks.append((f, 'M', s, 'Q-DAGFlow-noW|%g' % ds, 'Q-DAGFlow', dict(ek, budget_factor=None, qd_kw={'witness': False})))
    sweep(tasks, 'data/exp_practice.json')

ABL = {
    'Full': {},
    'w/o hedging': {'hedge': False},
    'w/o LP-MA': {'use_lp': False},
    'w/o path-risk': {'risk': False},
    'w/o switching': {'switch': False},
    'w/o witness/CPD': {'witness': False},
    'w/o reconfig': {'reconfig': False},
    'w/o LCB': {'lcb': False},
    'w/o emulation': {'emulation': False},
}

def ablation():
    tasks = []
    for f in FAMS:
        for s in range(1, 31):
            for lab, kw in ABL.items():
                tasks.append((f, 'M', s, lab, 'Q-DAGFlow', {'budget_factor': None, 'qd_kw': kw}))
    sweep(tasks, 'data/exp_ablation.json')

def budget():
    tasks = []
    for f in ['VQA-Loop', 'SQD-Hybrid', 'Epigenomics-H']:
        for s in range(1, 16):
            for bf in [0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]:
                tasks.append((f, 'M', s, 'QD|%g' % bf, 'Q-DAGFlow', {'budget_factor': bf}))
            tasks.append((f, 'M', s, 'QD|inf', 'Q-DAGFlow', {'budget_factor': None}))
            for b in BASE:
                tasks.append((f, 'M', s, b, b, C))
    sweep(tasks, 'data/exp_budget.json')

def sens():
    tasks = []
    meths = [('HEFT-Q', 'HEFT-Q', C), ('PEFT-Q', 'PEFT-Q', C), ('QG-Dyn', 'QG-Dyn', C), ('Q-DAGFlow', 'Q-DAGFlow', QD)]
    for f in ['VQA-Loop', 'SQD-Hybrid', 'Epigenomics-H']:
        for s in range(1, 16):
            for qs in [0.25, 0.5, 1.0, 2.0, 4.0]:
                for lab, m, kw in meths:
                    tasks.append((f, 'M', s, '%s|queue|%g' % (lab, qs), m, dict(kw, plat_kw={'queue_scale': qs})))
            for mn in [16, 32, 64, 128, 256, 500]:
                for lab, m, kw in meths:
                    tasks.append((f, 'M', s, '%s|nodes|%d' % (lab, mn), m, dict(kw, m_nodes=mn)))
            for ql in [0.5, 0.7, 0.9, 1.2]:
                tasks.append((f, 'M', s, 'Q-DAGFlow|sigq|%g' % ql, 'Q-DAGFlow', dict(QD, plat_kw={'sig_q': ql})))
                tasks.append((f, 'M', s, 'QG-Dyn|sigq|%g' % ql, 'QG-Dyn', dict(C, plat_kw={'sig_q': ql})))
            for cl in [1.0, 5.0, 30.0, 120.0, 600.0]:
                tasks.append((f, 'M', s, 'Q-DAGFlow|cancel|%g' % cl, 'Q-DAGFlow', dict(QD, plat_kw={'cancel_lat': cl})))
    sweep(tasks, 'data/exp_sens.json')

def ablation_L():
    tasks = []
    for f in ['VQA-Loop', 'Epigenomics-H', '1000Genome-H', 'SQD-Hybrid']:
        for s in range(1, 11):
            for lab in ['Full', 'w/o hedging', 'w/o LP-MA', 'w/o switching']:
                tasks.append((f, 'L', s, lab, 'Q-DAGFlow', {'budget_factor': None, 'qd_kw': ABL[lab]}))
            tasks.append((f, 'L', s, 'QG-Dyn', 'QG-Dyn', C))
    sweep(tasks, 'data/exp_ablation_L.json')

def scal():
    tasks = []
    for f in ['QK-HPO', 'VQA-Loop', 'SQD-Hybrid', 'Montage-H']:
        for sc in ['S', 'M', 'L', 'XL', 'XXL']:
            for s in range(1, 4):
                for lab, m, kw in [('HEFT-Q', 'HEFT-Q', C), ('PEFT-Q', 'PEFT-Q', C), ('QG-Dyn', 'QG-Dyn', C), ('Q-DAGFlow', 'Q-DAGFlow', QD)]:
                    tasks.append((f, sc, s, lab, m, kw))
    sweep(tasks, 'data/exp_scal.json', procs=1)

def strong():
    """Strengthened baselines: queue-greedy with hedged modes, and fidelity-latency scalarisation (three weights)."""
    tasks = []
    for sc in ['S', 'M', 'L']:
        for f in FAMS:
            for s in range(1, 31):
                tasks.append((f, sc, s, 'QGH-Dyn', 'QGH-Dyn', dict(C)))
                for w in (0.25, 0.5, 0.75):
                    tasks.append((f, sc, s, 'FT-Dyn|%g' % w, 'FT-Dyn', dict(C, qd_kw={'w': w})))
    sweep(tasks, 'data/exp_strong.json')

if __name__ == '__main__':
    globals()[sys.argv[1]]()
