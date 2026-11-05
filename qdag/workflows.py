"""Hybrid quantum-classical AI workflow generators.

Three application families are modelled after published hybrid patterns (quantum-kernel model
selection, variational training loops, sample-based quantum diagonalisation); three scientific
workflow families use WfCommons recipes (Montage, Epigenomics, 1000Genome) derived from real
execution traces, hybridised by inserting quantum post-processing stages.
"""
import math, warnings
import numpy as np

warnings.filterwarnings('ignore')
MB = 1e6


class Workflow:
    def __init__(self, name):
        self.name = name
        self.dur, self.q, self.label = [], [], []
        self.succ, self.pred = [], []
        self.ebytes = {}

    def add(self, label, dur=0.0, q=None):
        j = len(self.dur)
        self.dur.append(float(dur)); self.q.append(q); self.label.append(label)
        self.succ.append([]); self.pred.append([])
        return j

    def edge(self, i, j, nbytes):
        if j in self.succ[i]:
            return
        self.succ[i].append(j); self.pred[j].append(i); self.ebytes[(i, j)] = float(nbytes)

    def finalize(self):
        self.n = len(self.dur)
        self.dur = np.array(self.dur)
        self.isq = np.array([q is not None for q in self.q])
        self.topo = self._topo()
        self.n_edges = len(self.ebytes)
        return self

    def _topo(self):
        indeg = [len(p) for p in self.pred]
        order = [j for j in range(len(indeg)) if indeg[j] == 0]
        k = 0
        while k < len(order):
            i = order[k]; k += 1
            for j in self.succ[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    order.append(j)
        assert len(order) == len(indeg), 'cycle'
        return order


def qtask(rng, cls, n, **kw):
    if cls == 'K':
        br = bc = kw.get('b', 16)
        nc = kw.get('nc', br * bc)
        return dict(cls='K', n=n, nc=nc, br=br, bc=bc, eps=kw.get('eps', 0.08), kappa=0.5)
    if cls == 'V':
        P = kw.get('P', n)
        return dict(cls='V', n=n, nc=2 * P + 1, P=P, eps=kw.get('eps', 0.05), kappa=0.3)
    return dict(cls='S', n=n, nc=kw.get('nc', 2), nv=kw.get('nv', 100_000), eps=None, kappa=None)


def ln(rng, mean, cv=0.3):
    s = math.sqrt(math.log(1 + cv * cv)); return float(mean * math.exp(rng.normal(-s * s / 2, s)))


# ------------------------------------------------------------------------------ QK-HPO
def qk_hpo(rng, scale):
    H, G, F, S = {'XS': (2, 3, 2, 2), 'S': (6, 6, 4, 4), 'M': (20, 10, 5, 8), 'L': (44, 14, 5, 16), 'XL': (60, 16, 5, 16), 'XXL': (100, 18, 5, 16)}[scale]
    w = Workflow('QK-HPO')
    ing = w.add('ingest', ln(rng, 300)); cln = w.add('clean', ln(rng, 240)); w.edge(ing, cln, 2000 * MB)
    scores = []
    for c in range(H):
        n = int(rng.choice([16, 20, 22, 24, 26, 28]))
        fe = w.add('featurize', ln(rng, 150)); w.edge(cln, fe, 800 * MB)
        blocks = []
        for a in range(G):
            for b in range(a, G):
                k = w.add('kblock', 0.0, qtask(rng, 'K', n, nc=(16 * 15 // 2 if a == b else 256)))
                w.edge(fe, k, 1 * MB); blocks.append(k)
        asm = w.add('assemble', ln(rng, 40))
        for k in blocks:
            w.edge(k, asm, 0.01 * MB)
        sc = w.add('score', ln(rng, 20))
        for f in range(F):
            cv = w.add('svm_cv', ln(rng, 90)); w.edge(asm, cv, 5 * MB); w.edge(cv, sc, 0.1 * MB)
            cb = w.add('classical_cv', ln(rng, 600, 0.5)); w.edge(fe, cb, 300 * MB); w.edge(cb, sc, 0.1 * MB)
        scores.append(sc)
    sel = w.add('select', ln(rng, 15))
    for s in scores:
        w.edge(s, sel, 0.1 * MB)
    fin = w.add('final_train', ln(rng, 400)); w.edge(sel, fin, 1 * MB)
    n_best = 24
    tb = []
    for a in range(max(2, G // 2)):
        k = w.add('test_kblock', 0.0, qtask(rng, 'K', n_best)); w.edge(fin, k, 1 * MB); tb.append(k)
    rep = w.add('report', ln(rng, 30))
    for s in range(S):
        pr = w.add('predict', ln(rng, 80))
        for k in tb:
            w.edge(k, pr, 0.01 * MB)
        w.edge(pr, rep, 10 * MB)
    return w.finalize()


# ------------------------------------------------------------------------------ VQA-Loop
def vqa_loop(rng, scale):
    R, I = {'XS': (3, 4), 'S': (6, 14), 'M': (24, 20), 'L': (60, 38), 'XL': (100, 40), 'XXL': (160, 50)}[scale]
    w = Workflow('VQA-Loop')
    ld = w.add('load', ln(rng, 200))
    evals = []
    for r in range(R):
        n = int(rng.choice([16, 20, 24, 28, 32, 40, 48, 56]))
        ini = w.add('init', ln(rng, 60)); w.edge(ld, ini, 500 * MB)
        prev = ini
        for it in range(I):
            g = w.add('grad', 0.0, qtask(rng, 'V', n)); w.edge(prev, g, 0.01 * MB)
            u = w.add('update', ln(rng, 12)); w.edge(g, u, 0.01 * MB)
            prev = u
        ev = w.add('evaluate', ln(rng, 180)); w.edge(prev, ev, 0.01 * MB); w.edge(ld, ev, 500 * MB)
        evals.append(ev)
    agg = w.add('aggregate', ln(rng, 60))
    for e in evals:
        w.edge(e, agg, 1 * MB)
    w.edge(agg, w.add('model_select', ln(rng, 30)), 1 * MB)
    return w.finalize()


# ------------------------------------------------------------------------------ SQD-Hybrid
def sqd_hybrid(rng, scale):
    M, Rr, W, D = {'XS': (2, 2, 3, 3), 'S': (4, 3, 6, 6), 'M': (12, 4, 10, 10), 'L': (40, 5, 12, 12), 'XL': (60, 5, 12, 12), 'XXL': (120, 5, 12, 12)}[scale]
    w = Workflow('SQD-Hybrid')
    feats = []
    for mol in range(M):
        n = int(rng.choice([32, 40, 48, 56, 64, 72]))
        prep = w.add('hamiltonian', ln(rng, 300))
        prev = prep
        for r in range(Rr):
            smp = w.add('sample', 0.0, qtask(rng, 'S', n, nc=2)); w.edge(prev, smp, 1 * MB)
            rec = [w.add('recover', ln(rng, 150)) for _ in range(W)]
            for x in rec:
                w.edge(smp, x, 20 * MB)
            dg = [w.add('diagonalize', ln(rng, 1200, 0.5)) for _ in range(D)]
            for i, x in enumerate(dg):
                w.edge(rec[i % W], x, 200 * MB); w.edge(rec[(i + 1) % W], x, 200 * MB)
            mg = w.add('merge', ln(rng, 60))
            for x in dg:
                w.edge(x, mg, 50 * MB)
            prev = mg
        ft = w.add('features', ln(rng, 90)); w.edge(prev, ft, 10 * MB); feats.append(ft)
    tr = w.add('surrogate_train', ln(rng, 1800))
    for f in feats:
        w.edge(f, tr, 10 * MB)
    w.edge(tr, w.add('validate', ln(rng, 300)), 100 * MB)
    return w.finalize()


# ------------------------------------------------------------------------------ WfCommons hybrids
def wfcommons_hybrid(rng, scale, recipe, cat, phi=0.3):
    from wfcommons import WorkflowGenerator
    from wfcommons.wfchef.recipes import MontageRecipe, EpigenomicsRecipe, GenomeRecipe
    R = {'Montage': MontageRecipe, 'Epigenomics': EpigenomicsRecipe, '1000Genome': GenomeRecipe}[recipe]
    ntask = {'XS': 60, 'S': 200, 'M': 1000, 'L': 4000, 'XL': 8000, 'XXL': 16000}[scale]
    import random
    random.seed(int(rng.integers(1 << 30))); np.random.seed(int(rng.integers(1 << 30)))
    g = WorkflowGenerator(R.from_num_tasks(ntask)).build_workflow()
    names = list(g.nodes)
    idx = {nm: i for i, nm in enumerate(names)}
    w = Workflow(recipe + '-H')
    for nm in names:
        t = g.nodes[nm]['task']
        w.add(nm.rsplit('_', 1)[0], max(0.5, float(t.runtime)))
    outs = {nm: {f.file_id if hasattr(f, 'file_id') else f.name: f.size for f in g.nodes[nm]['task'].output_files} for nm in names}
    for u, v in g.edges:
        ins = {f.file_id if hasattr(f, 'file_id') else f.name for f in g.nodes[v]['task'].input_files}
        nb = sum(sz for fid, sz in outs[u].items() if fid in ins)
        w.edge(idx[u], idx[v], max(nb, 1e3))
    # hybridisation: attach quantum post-processing stage to a fraction phi of tasks of category `cat`
    base_n = w.n if hasattr(w, 'n') else len(w.dur)
    cand = [j for j in range(base_n) if w.label[j] == cat]
    for j in cand:
        if rng.random() > phi:
            continue
        cls = rng.choice(['K', 'V', 'S'], p=[0.4, 0.3, 0.3])
        if cls == 'K':
            q = qtask(rng, 'K', int(rng.choice([16, 20, 24, 28])), b=12, nc=144)
        elif cls == 'V':
            q = qtask(rng, 'V', int(rng.choice([20, 28, 36, 48])), P=12)
        else:
            q = qtask(rng, 'S', int(rng.choice([32, 48, 64])), nc=1, nv=50_000)
        k = w.add('q_' + cat, 0.0, q)
        succs = list(w.succ[j])
        w.edge(j, k, 5 * MB)
        for s in succs:
            w.edge(k, s, 0.1 * MB)
    return w.finalize()


FAMILIES = {
    'QK-HPO': qk_hpo,
    'VQA-Loop': vqa_loop,
    'SQD-Hybrid': sqd_hybrid,
    'Montage-H': lambda rng, sc: wfcommons_hybrid(rng, sc, 'Montage', 'mBackground', phi=0.8),
    'Epigenomics-H': lambda rng, sc: wfcommons_hybrid(rng, sc, 'Epigenomics', 'map', phi=0.5),
    '1000Genome-H': lambda rng, sc: wfcommons_hybrid(rng, sc, '1000Genome', 'frequency', phi=0.5),
}


def make(family, scale, seed):
    import zlib
    rng = np.random.default_rng(seed * 31 + zlib.crc32(family.encode()) % 1000)
    return FAMILIES[family](rng, scale)
