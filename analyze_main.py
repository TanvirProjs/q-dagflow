import json, numpy as np, pandas as pd
from scipy import stats
from figstyle import *
setup()
D = 'data/'
FAMS = ['QK-HPO', 'VQA-Loop', 'SQD-Hybrid', 'Montage-H', 'Epigenomics-H', '1000Genome-H']
BASE = ['HEFT-Q', 'PEFT-Q', 'SHEFT-Q', 'QG-Dyn', 'FG-Dyn', 'CF-Dyn']
METH = BASE + ['Q-DAGFlow']
df = pd.DataFrame([r for r in json.load(open(D + 'exp_main.json')) if 'error' not in r])
df.loc[(df.method != 'LB') & (~df.complete.astype(bool)), 'makespan'] = np.inf
lb = df[df.method == 'LB'].set_index(['family', 'scale', 'seed']).makespan
d = df[df.method != 'LB'].copy()
d['lb'] = [lb.get((f, s, e), np.nan) for f, s, e in zip(d.family, d.scale, d.seed)]
d['norm'] = d.makespan / d.lb
out = {}

def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum(); lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))

rows = []; tests = []
for f in FAMS:
    for sc in ['S', 'M', 'L']:
        sub = d[(d.family == f) & (d.scale == sc)]
        if len(sub) == 0: continue
        piv = sub.pivot_table(index='seed', columns='method', values='makespan', aggfunc='first').dropna()
        row = {'family': f, 'scale': sc, 'n_tasks': int(sub.n_tasks.median()), 'n_q': int(sub.n_q.median()), 'N': len(piv)}
        for m in METH:
            x = piv[m].values
            row[m + '_mean'] = np.mean(x) if np.all(np.isfinite(x)) else np.inf
            row[m + '_sd'] = np.std(x, ddof=1) if np.all(np.isfinite(x)) else np.nan
            row[m + '_p95'] = np.percentile(x, 95) if np.all(np.isfinite(x)) else np.inf
        bb = min(BASE, key=lambda m: row[m + '_mean'])
        row['best_base'] = bb
        row['impr_best'] = 100 * (1 - row['Q-DAGFlow_mean'] / row[bb + '_mean'])
        row['impr_heft'] = 100 * (1 - row['Q-DAGFlow_mean'] / row['HEFT-Q_mean'])
        rows.append(row)
        for m in BASE:
            a, b = piv['Q-DAGFlow'].values, piv[m].values
            ok = np.isfinite(b)
            if ok.sum() < len(b):   # DNF counts as Q-DAGFlow win
                b = np.where(ok, b, 1e12)
            try:
                p = stats.wilcoxon(a, b, alternative='two-sided').pvalue
            except ValueError:
                p = 1.0
            tests.append(dict(family=f, scale=sc, base=m, p=p, delta=cliffs(b, a), win=int((a < b).sum()), loss=int((a > b).sum()),
                              ratio=float(np.exp(np.mean(np.log(np.minimum(a / b, 1e3)))))))
T = pd.DataFrame(rows)
W = pd.DataFrame(tests)
# Holm correction over all tests
order = np.argsort(W.p.values); m_ = len(W); adj = np.empty(m_); run = 0
for k, i in enumerate(order):
    run = max(run, min(1.0, (m_ - k) * W.p.values[i])); adj[i] = run
W['p_holm'] = adj
W['sig'] = np.where(W.p_holm < 0.05, np.where(W.win > W.loss, 'better', 'worse'), 'n.s.')
T.to_csv(D + 'main_table.csv', index=False); W.to_csv(D + 'main_tests.csv', index=False)
summ = W.groupby('base').agg(better=('sig', lambda s: (s == 'better').sum()), ns=('sig', lambda s: (s == 'n.s.').sum()),
                             worse=('sig', lambda s: (s == 'worse').sum()), med_delta=('delta', 'median'),
                             gm_ratio=('ratio', lambda r: float(np.exp(np.mean(np.log(r))))), min_p=('p_holm', 'min'))
print(summ.round(3).to_string())
# Friedman over (family, scale, seed) blocks
P = d.pivot_table(index=['family', 'scale', 'seed'], columns='method', values='makespan', aggfunc='first')[METH]
R = P.rank(axis=1, method='average')
fr = stats.friedmanchisquare(*[P[m].replace(np.inf, 1e15).values for m in METH])
avg_rank = R.mean().sort_values()
k, N = len(METH), len(P)
CD = 2.949 * np.sqrt(k * (k + 1) / (6 * N))
print('Friedman chi2=%.1f p=%.2e N=%d CD=%.3f' % (fr.statistic, fr.pvalue, N, CD)); print(avg_rank.round(3).to_string())
json.dump(dict(friedman=fr.statistic, friedman_p=fr.pvalue, N=N, CD=CD, ranks=avg_rank.to_dict(),
               summary=summ.reset_index().to_dict(orient='records')), open(D + 'main_stats.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 250)
print(T[['family', 'scale', 'n_tasks', 'n_q', 'best_base', 'impr_best', 'impr_heft']].round(1).to_string())
# extra aggregates
agg = d.groupby('method')[['qpu_cost', 'viol_mean', 'util', 'sched_time', 'qlat_mean', 'qlat_p95', 'n_hedged']].mean()
print(agg.round(3).to_string())
