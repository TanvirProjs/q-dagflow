"""Robustness, uncertainty and operational analyses of the main experiment (Section 6.1-6.2), and the
strengthened-baseline comparison (Section 6.3). Writes data/values_robust.json."""
import json, sys, numpy as np, pandas as pd
from scipy import stats
D = 'data/'
FAMS = ['QK-HPO', 'VQA-Loop', 'SQD-Hybrid', 'Montage-H', 'Epigenomics-H', '1000Genome-H']
BASE = ['HEFT-Q', 'PEFT-Q', 'SHEFT-Q', 'QG-Dyn', 'FG-Dyn', 'CF-Dyn']
COMP = [b for b in BASE if b != 'FG-Dyn']
V = {}
rng = np.random.default_rng(12345)
df = pd.DataFrame([r for r in json.load(open(D + 'exp_main.json')) if 'error' not in r])
d = df[df.method != 'LB']
T = pd.read_csv(D + 'main_table.csv')
MS = d.pivot_table(index=['family', 'scale', 'seed'], columns='method', values='makespan')


def boot_delta(qd, bb, B=4000):
    """95% bootstrap CI (paired resampling of instances) of 100*(1 - mean(qd)/mean(bb))."""
    n = len(qd); idx = rng.integers(0, n, size=(B, n))
    r = 100 * (1 - qd[idx].mean(axis=1) / bb[idx].mean(axis=1))
    return np.percentile(r, [2.5, 97.5])


# ---- 1. bootstrap CIs for the Delta column of Table 5
ci = {}
for _, r in T.iterrows():
    x = MS.loc[(r.family, r.scale)]
    lo, hi = boot_delta(x['Q-DAGFlow'].values, x[r.best_base].values)
    ci['%s|%s' % (r.family, r.scale)] = [round(lo, 1), round(hi, 1)]
json.dump(ci, open(D + 'main_delta_ci.json', 'w'), indent=1)
V['N_CI_POS'] = str(sum(1 for v in ci.values() if v[0] > 0))
V['N_CI_ZERO'] = str(sum(1 for v in ci.values() if v[0] <= 0 <= v[1]))
V['N_CI_NEG'] = str(sum(1 for v in ci.values() if v[1] < 0))

# ---- 2. instance-level robustness per family (pooled over scales)
TIE = 0.001
rows = []
allr, vb_all = [], []
for f in FAMS:
    rr, vb, p95 = [], [], []
    for sc in 'SML':
        bb = T[(T.family == f) & (T.scale == sc)].best_base.iloc[0]
        x = MS.loc[(f, sc)]
        rr += list(x['Q-DAGFlow'] / x[bb])
        vb += list(x['Q-DAGFlow'] / x[COMP].min(axis=1))
        p95.append(np.percentile(x['Q-DAGFlow'], 95) / np.percentile(x[bb], 95))
    rr, vb = np.array(rr), np.array(vb)
    allr += list(rr); vb_all += list(vb)
    rows.append(dict(family=f, win=int((rr < 1 - TIE).sum()), tie=int((abs(rr - 1) <= TIE).sum()), loss=int((rr > 1 + TIE).sum()),
                     med=np.median(rr), p5=np.percentile(rr, 5), p95=np.percentile(rr, 95), worst=rr.max(),
                     vb1=100 * np.mean(vb <= 1.01), p95chg=100 * (np.mean(p95) - 1)))
R = pd.DataFrame(rows); R.to_csv(D + 'robust_table.csv', index=False)
allr, vb_all = np.array(allr), np.array(vb_all)
V['ROB_WIN'] = str(int((allr < 1 - TIE).sum())); V['ROB_TIE'] = str(int((abs(allr - 1) <= TIE).sum())); V['ROB_LOSS'] = str(int((allr > 1 + TIE).sum()))
V['ROB_LOSS5'] = str(int((allr > 1.05).sum())); V['ROB_WORST'] = '%.1f' % (100 * (allr.max() - 1))
wf = R.set_index('family').worst.idxmax(); V['ROB_WORST_FAM'] = wf
V['ROB_VB1'] = '%.0f' % (100 * np.mean(vb_all <= 1.01))
lines = [': **Table 8.** Instance-level robustness of Q-DAGFlow (90 paired instances per family, scales S–L pooled). Ratio: Q-DAGFlow makespan / makespan of the best baseline of the instance\'s cell (Table 5); wins, ties and losses use a 0.1% tolerance. "≤ VB + 1%": share of instances within 1% of the virtual best, i.e. the fastest of the five competitive baselines *on that instance*. p95 change: change of the 95th-percentile makespan across instances, relative to the cell\'s best baseline (mean over scales; negative = lighter tail).', '',
         '| Family | Wins / ties / losses | Median ratio | 5th–95th pct. ratio | Worst ratio | ≤ VB + 1% | p95 change |', '|:--|:-:|:-:|:-:|:-:|:-:|:-:|']
for _, r in R.iterrows():
    lines.append('| %s | %d / %d / %d | %.3f | %.3f–%.3f | %.3f | %.0f%% | %+.1f%% |' % (r.family, r.win, r.tie, r.loss, r.med, r.p5, r.p95, r.worst, r.vb1, r.p95chg))
V['TABLE_ROBUST'] = '\n'.join(l.replace('| -', '| −') for l in lines)
for _, r in R.iterrows():
    k = r.family.split('-')[0].upper()
    V['ROB_P95_%s' % k] = '%.1f' % r.p95chg; V['ROB_WORST_%s' % k] = '%.1f' % (100 * (r.worst - 1))
    V['ROB_LOSSES_%s' % k] = str(r.loss)
V['ROB_P95_MIN'] = '%.1f' % R.p95chg.min()
for _, r in R.iterrows():
    V['ROB_P95ABS_%s' % r.family.split('-')[0].upper()] = '%.1f' % abs(r.p95chg)

# ---- 3. operational profile per family (Q-DAGFlow vs QG-Dyn; all scales)
def gm(a):
    a = np.asarray(a, float); a = a[np.isfinite(a) & (a > 0)]
    return float(np.exp(np.mean(np.log(a)))) if len(a) else float('nan')
op = []
key = ['family', 'scale', 'seed']
P = {m: d[d.method == m].set_index(key) for m in ['Q-DAGFlow', 'QG-Dyn']}
lat_r, p95_r = [], []
for f in FAMS:
    q, g = P['Q-DAGFlow'].loc[f], P['QG-Dyn'].loc[f]
    both = (q.qlat_mean > 0) & (g.qlat_mean > 0)
    lr = (q.qlat_mean / g.qlat_mean)[both]; pr = (q.qlat_p95 / g.qlat_p95)[both]
    lat_r += list(lr); p95_r += list(pr)
    vb = d[(d.family == f) & d.method.isin(COMP)].viol_mean.mean()
    op.append(dict(family=f, emul_qd=100 * q.n_emul.sum() / q.n_q.sum(), emul_qg=100 * g.n_emul.sum() / g.n_q.sum(),
                   hedged=100 * q.n_hedged.sum() / max(1, (q.n_qpu - q.n_double).sum()) if q.n_qpu.sum() else float('nan'),
                   lat=100 * (gm(lr) - 1) if both.any() else float('nan'), latp95=100 * (gm(pr) - 1) if both.any() else float('nan'),
                   cost=100 * (q.qpu_cost.sum() / g.qpu_cost.sum() - 1), viol_qd=q.viol_mean.mean(), viol_base=vb))
O = pd.DataFrame(op); O.to_csv(D + 'operational_table.csv', index=False)
fmt = lambda v, s='%+.0f%%': '–' if not np.isfinite(v) else s % v
lines = [': **Table 6.** Operational profile (90 instances per family, scales S–L pooled). Emulated: share of quantum tasks executed by classical emulation. Hedged: share of Q-DAGFlow\'s QPU jobs submitted as hedged pairs. Latency changes are geometric means of the paired per-instance ratio Q-DAGFlow / QG-Dyn over instances in which both submit QPU jobs; spend is the change in total billed QPU time. Violation: mean per-task probability of missing the precision target, for Q-DAGFlow and averaged over the five competitive baselines.', '',
         '| Family | Emulated (QD / QG) | Hedged | Mean QPU latency | p95 QPU latency | QPU spend | Violation (QD / baselines) |', '|:--|:-:|:-:|:-:|:-:|:-:|:-:|']
for _, r in O.iterrows():
    lines.append('| %s | %.0f%% / %.0f%% | %s | %s | %s | %s | %.3f / %.3f |' % (r.family, r.emul_qd, r.emul_qg, fmt(r.hedged, '%.0f%%'), fmt(r.lat), fmt(r.latp95),
                 ('–' if not np.isfinite(r.cost) else ('%+.0f%%' % r.cost if r.family != 'QK-HPO' else '−100%')), r.viol_qd, r.viol_base))
V['TABLE_OPS'] = '\n'.join(l.replace('| -', '| −').replace(' -', ' −') if l.startswith('| ') else l for l in lines)
V['QLAT_GM_RED'] = '%.0f' % (100 * (1 - gm(lat_r))); V['QLAT_P95_GM_RED'] = '%.0f' % (100 * (1 - gm(p95_r)))
Ow = O[O.family != 'QK-HPO'].set_index('family')
V['QLAT_RED_MIN'] = '%.0f' % (-Ow.lat.max()); V['QLAT_RED_MIN_FAM'] = Ow.lat.idxmax()
V['QLAT_RED_MAX'] = '%.0f' % (-Ow.lat.min()); V['QLAT_RED_MAX_FAM'] = Ow.lat.idxmin()
V['COST_SQD_CHG'] = '%.1f' % abs(Ow.loc['SQD-Hybrid', 'cost']); V['COST_VQA_CHG'] = '%.0f' % abs(Ow.loc['VQA-Loop', 'cost'])
assert Ow.loc['SQD-Hybrid', 'cost'] > 0 and Ow.loc['VQA-Loop', 'cost'] < 0
V['COST_WF_MIN'] = '%.0f' % (-Ow.loc[['Montage-H', 'Epigenomics-H', '1000Genome-H'], 'cost'].max())
V['COST_WF_MAX'] = '%.0f' % (-Ow.loc[['Montage-H', 'Epigenomics-H', '1000Genome-H'], 'cost'].min())
q = d[(d.family == 'QK-HPO')]
V['QK_BASE_QPU_MAX'] = '%.1f' % (100 * max(q[q.method == m].n_qpu.sum() / q[q.method == m].n_q.sum() for m in COMP))
V['QK_EMUL_MIN'] = '%.1f' % (100 - float(V['QK_BASE_QPU_MAX']))
# iso-precision check in the main experiment
vio = d.groupby('method').viol_mean.mean()
V['VIOL_MAIN_QD'] = '%.3f' % vio['Q-DAGFlow']
V['VIOL_MAIN_BMIN'] = '%.3f' % min(vio[b] for b in COMP); V['VIOL_MAIN_BMAX'] = '%.3f' % max(vio[b] for b in COMP)
V['VIOL_MAIN_FG'] = '%.3f' % vio['FG-Dyn']
fv = d.pivot_table(index='family', columns='method', values='viol_mean')
V['VIOL_HIGHER_FAMS'] = ' and '.join(f for f in FAMS if fv.loc[f, 'Q-DAGFlow'] > fv.loc[f, COMP].min() + 1e-9 and fv.loc[f, 'Q-DAGFlow'] > fv.loc[f, COMP].mean())
# FG-Dyn slowdown
fgr = (MS['FG-Dyn'] / MS['Q-DAGFlow']).replace(np.inf, np.nan)
g = np.exp(np.log(fgr).groupby(['family']).mean())
V['FG_SLOW_QV_MIN'] = '%.1f' % g[['VQA-Loop', 'SQD-Hybrid']].min(); V['FG_SLOW_QV_MAX'] = '%.1f' % g[['VQA-Loop', 'SQD-Hybrid']].max()
V['FG_SLOW_WF_MIN'] = '%.0f' % g[['Montage-H', 'Epigenomics-H', '1000Genome-H']].min(); V['FG_SLOW_WF_MAX'] = '%.0f' % g[['Montage-H', 'Epigenomics-H', '1000Genome-H']].max()
V['FG_DNF'] = str(int(np.isinf(MS['FG-Dyn']).sum()))

# ---- 4. strengthened baselines
try:
    st = pd.DataFrame([r for r in json.load(open(D + 'exp_strong.json')) if 'error' not in r])
except FileNotFoundError:
    st = None
if st is not None:
    SM = st.pivot_table(index=['family', 'scale', 'seed'], columns='label', values='makespan')
    ws = [c for c in SM.columns if c.startswith('FT-Dyn|')]
    rows = []; ntest = 0; pv = {}
    for f in FAMS:
        for sc in 'SML':
            x = MS.loc[(f, sc)]; s = SM.loc[(f, sc)]
            # oracle-tuned weight: best mean makespan over the three weights (tuned on the test instances)
            wbest = min(ws, key=lambda c: s[c].mean())
            for lab, col in [('QGH-Dyn', s['QGH-Dyn']), ('FT-Dyn*', s[wbest])]:
                a, b = x['Q-DAGFlow'].values, col.reindex(x.index).values
                try:
                    p = stats.wilcoxon(a, b).pvalue
                except ValueError:
                    p = 1.0
                pv[(f, sc, lab)] = p
                rows.append(dict(family=f, scale=sc, base=lab, w=wbest.split('|')[1] if lab == 'FT-Dyn*' else '', qd=a.mean(), bm=b.mean(),
                                 impr=100 * (1 - a.mean() / b.mean()), better=np.mean(a < b), viol_b=st[(st.family == f) & (st.scale == sc) & (st.label == (wbest if lab == 'FT-Dyn*' else 'QGH-Dyn'))].viol_mean.mean()))
    S = pd.DataFrame(rows)
    # Holm over the 36 tests
    order = sorted(pv, key=pv.get); m = len(order); adj = {}; run = 0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pv[k])); adj[k] = run
    S['p_holm'] = [adj[(r.family, r.scale, r.base)] for _, r in S.iterrows()]
    S['sig'] = ['better' if (r.p_holm < 0.05 and r.impr > 0) else ('worse' if (r.p_holm < 0.05 and r.impr < 0) else 'ns') for _, r in S.iterrows()]
    S.to_csv(D + 'strong_table.csv', index=False)
    lines = [': **Table 9.** Q-DAGFlow against strengthened baselines (30 paired instances per cell). QGH-Dyn: queue-greedy routing with the same hedged composite modes as Q-DAGFlow. FT-Dyn*: fidelity–latency scalarisation in the style of multi-objective quantum-cloud schedulers, with the weight $w\\in\\{0.25,0.5,0.75\\}$ chosen per cell on the test instances themselves (an optimistic tuning). Δ: change of Q-DAGFlow\'s mean makespan relative to the baseline (positive = faster); markers as in Table 5 (Holm correction over the 36 tests).', '',
             '| Family | Scale | QGH-Dyn (h) | Δ vs QGH-Dyn (%) | FT-Dyn* (h) | $w$ | Δ vs FT-Dyn* (%) | Q-DAGFlow (h) |', '|:--|:-:|--:|--:|--:|:-:|--:|--:|']
    mk = lambda s: '▲' if s == 'better' else ('▼' if s == 'worse' else '○')
    for f in FAMS:
        for sc in 'SML':
            a = S[(S.family == f) & (S.scale == sc) & (S.base == 'QGH-Dyn')].iloc[0]; b = S[(S.family == f) & (S.scale == sc) & (S.base == 'FT-Dyn*')].iloc[0]
            lines.append('| %s | %s | %.2f | %+.1f %s | %.2f | %s | %+.1f %s | %.2f |' % (f, sc, a.bm / 3600, a.impr, mk(a.sig), b.bm / 3600, b.w, b.impr, mk(b.sig), a.qd / 3600))
    V['TABLE_STRONG'] = '\n'.join(l.replace('| -', '| −') for l in lines)
    for lab, k in [('QGH-Dyn', 'QGH'), ('FT-Dyn*', 'FT')]:
        s = S[S.base == lab]
        V['STR_%s_BETTER' % k] = str(int((s.sig == 'better').sum())); V['STR_%s_WORSE' % k] = str(int((s.sig == 'worse').sum())); V['STR_%s_NS' % k] = str(int((s.sig == 'ns').sum()))
        V['STR_%s_MAX' % k] = '%.1f' % s.impr.max(); V['STR_%s_MIN' % k] = '%.1f' % s.impr.min()
        allq = MS['Q-DAGFlow']; col = SM['QGH-Dyn'] if lab == 'QGH-Dyn' else None
    # geometric-mean ratios
    q = MS['Q-DAGFlow']
    V['STR_QGH_GM'] = '%.3f' % gm((q / SM['QGH-Dyn'].reindex(q.index)).values)
    qg = MS['QG-Dyn']
    V['STR_QGH_VS_QG_GM'] = '%.3f' % gm((SM['QGH-Dyn'].reindex(qg.index) / qg).values)
    for f in ['VQA-Loop', 'SQD-Hybrid', 'Montage-H', 'Epigenomics-H', '1000Genome-H']:
        k = f.split('-')[0].upper()
        s = S[(S.family == f) & (S.base == 'QGH-Dyn')]
        V['STR_QGH_%s' % k] = '%.1f–%.1f' % (s.impr.min(), s.impr.max()) if s.impr.min() != s.impr.max() else '%.1f' % s.impr.min()
        s2 = S[(S.family == f) & (S.base == 'FT-Dyn*')]
        V['STR_FT_%s' % k] = '%.1f–%.1f' % (s2.impr.min(), s2.impr.max())
    for f, sc in [('VQA-Loop', 'L'), ('SQD-Hybrid', 'L'), ('Epigenomics-H', 'L'), ('1000Genome-H', 'L'), ('VQA-Loop', 'M'), ('SQD-Hybrid', 'M'), ('Montage-H', 'L')]:
        k = f.split('-')[0].upper()
        V['STR_QGH_%s_%s' % (k, sc)] = '%.1f' % S[(S.family == f) & (S.scale == sc) & (S.base == 'QGH-Dyn')].impr.iloc[0]
        V['STR_FT_%s_%s' % (k, sc)] = '%.1f' % S[(S.family == f) & (S.scale == sc) & (S.base == 'FT-Dyn*')].impr.iloc[0]
    V['STR_VIOL_QGH'] = '%.3f' % st[st.label == 'QGH-Dyn'].viol_mean.mean()
    s = S[(S.family == 'VQA-Loop') & (S.base == 'QGH-Dyn')]
    V['STR_QGH_VQA_ABSMAX'] = '%.1f' % s.impr.abs().max()
    V['STR_QGH_WORSE_TEXT'] = 'never' if int(V['STR_QGH_WORSE']) == 0 else 'in %s settings' % V['STR_QGH_WORSE']
    V['STR_QGH_VS_QG_GM_PCT'] = '%.0f' % (100 * (1 - float(V['STR_QGH_VS_QG_GM'])))
    gaps = []
    for f in ['Epigenomics-H', '1000Genome-H']:
        xq = d[(d.family == f) & (d.scale == 'L') & (d.method == 'Q-DAGFlow')]
        for src in (d[(d.family == f) & (d.scale == 'L') & (d.method == 'QG-Dyn')], st[(st.family == f) & (st.scale == 'L') & (st.label == 'QGH-Dyn')]):
            gaps.append(100 * (xq.n_emul.sum() / xq.n_q.sum() - src.n_emul.sum() / src.n_q.sum()))
    V['STR_EMUL_GAP_MIN'] = '%.0f' % min(gaps); V['STR_EMUL_GAP_MAX'] = '%.0f' % max(gaps)
    print(S.round(3).to_string())
json.dump(V, open(D + 'values_robust.json', 'w'), indent=1, ensure_ascii=False)
print(R.round(3).to_string()); print(O.round(3).to_string())
print({k: v for k, v in V.items() if not k.startswith('TABLE')})
