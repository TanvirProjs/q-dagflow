import os
"""Collect every number quoted in the manuscript from the result files (single source of truth)."""
import json, math, numpy as np, pandas as pd
D = 'data/'
FAMS = ['QK-HPO', 'VQA-Loop', 'SQD-Hybrid', 'Montage-H', 'Epigenomics-H', '1000Genome-H']
BASE = ['HEFT-Q', 'PEFT-Q', 'SHEFT-Q', 'QG-Dyn', 'FG-Dyn', 'CF-Dyn']
METH = BASE + ['Q-DAGFlow']
V = {}
T = pd.read_csv(D + 'main_table.csv'); W = pd.read_csv(D + 'main_tests.csv'); st = json.load(open(D + 'main_stats.json'))
df = pd.DataFrame(json.load(open(D + 'exp_main.json')))
d = df[df.method != 'LB']
V['N_INST_TOTAL'] = '540'
V['N_SCHEDULES'] = '{:,}'.format(len(d))
ranks = st['ranks']; order = sorted(ranks, key=ranks.get)
V['RANK_QD'] = '%.2f' % ranks['Q-DAGFlow']; V['RANK_2ND'] = '%.2f' % ranks[order[1]]; V['RANK_2ND_NAME'] = order[1]
V['CD'] = '%.2f' % st['CD']; V['FRIEDMAN_CHI2'] = '%.1f' % st['friedman']
V['FRIEDMAN_EXP'] = str(int(math.floor(-math.log10(max(st['friedman_p'], 1e-300))))) if st['friedman_p'] > 0 else '300'
summ = {r['base']: r for r in st['summary']}
comp = [b for b in BASE if b != 'FG-Dyn']
V['SIG_BETTER_MIN'] = str(min(int(summ[b]['better']) for b in comp)); V['SIG_BETTER_MAX'] = str(max(int(summ[b]['better']) for b in comp))
V['SIG_WORSE_TOTAL'] = str(int(sum(summ[b]['worse'] for b in BASE)))
V['SIG_FG'] = str(int(summ['FG-Dyn']['better']))
_w = int(sum(summ[b]['worse'] for b in BASE))
V['SIG_WORSE_TEXT'] = 'none' if _w == 0 else str(_w)
for b in BASE:
    V['SIG_' + b.replace('-', '')] = str(int(summ[b]['better']))
four = [int(summ[b]['better']) for b in ['HEFT-Q', 'PEFT-Q', 'SHEFT-Q', 'FG-Dyn']]
V['SIG4_TEXT'] = 'in all 18 settings' if all(x == 18 for x in four) else 'in %d, %d, %d and %d of the 18 settings' % tuple(four)
V['MAX_IMPR_BEST'] = '%.1f' % T.impr_best.max()
V['MAX_IMPR_HEFT'] = '%.1f' % T.impr_heft.max()
V['MIN_IMPR_BEST'] = '%.1f' % T.impr_best.min()
def rng(f):
    x = T[T.family == f].impr_best.values; return '%.1f–%.1f%%' % (x.min(), x.max())
V['VQA_IMPR_RANGE'] = rng('VQA-Loop'); V['SQD_IMPR_RANGE'] = rng('SQD-Hybrid')
V['QK_IMPR_RANGE'] = rng('QK-HPO'); V['MON_IMPR_RANGE'] = rng('Montage-H'); V['EPI_IMPR_RANGE'] = rng('Epigenomics-H'); V['GEN_IMPR_RANGE'] = rng('1000Genome-H')
for f in FAMS:
    for sc in 'SML':
        r = T[(T.family == f) & (T.scale == sc)].iloc[0]
        V['IMPR_%s_%s' % (f.split('-')[0].upper(), sc)] = '%.1f' % r.impr_best
        V['IMPRH_%s_%s' % (f.split('-')[0].upper(), sc)] = '%.1f' % r.impr_heft
# QPU latency
nofg = d[d.method != 'FG-Dyn']
ql = nofg.groupby('method')[['qlat_mean', 'qlat_p95']].mean()
V['HEDGE_LAT_RED'] = '%.0f' % (100 * (1 - ql.loc['Q-DAGFlow', 'qlat_mean'] / ql.loc['QG-Dyn', 'qlat_mean']))
V['HEDGE_P95_RED'] = '%.0f' % (100 * (1 - ql.loc['Q-DAGFlow', 'qlat_p95'] / ql.loc['QG-Dyn', 'qlat_p95']))
V['QLAT_QD'] = '%.0f' % (ql.loc['Q-DAGFlow', 'qlat_mean'] / 60); V['QLAT_QG'] = '%.0f' % (ql.loc['QG-Dyn', 'qlat_mean'] / 60)
qd = d[d.method == 'Q-DAGFlow']
V['HEDGE_SHARE'] = '%.0f' % (100 * qd.n_hedged.sum() / max(1, (qd.n_qpu - qd.n_double).sum()))
V['DOUBLE_SHARE'] = '%.2f' % (100 * qd.n_double.sum() / max(1, qd.n_hedged.sum()))
cost = d.groupby('method').qpu_cost.mean()
assert cost['Q-DAGFlow'] < cost['QG-Dyn'] and cost['Q-DAGFlow'] < cost['HEFT-Q']
V['COST_QD_VS_QG'] = '%.0f' % (100 * (1 - cost['Q-DAGFlow'] / cost['QG-Dyn']))
V['COST_QD_VS_HEFT'] = '%.0f' % (100 * (1 - cost['Q-DAGFlow'] / cost['HEFT-Q']))
util = d.groupby('method').util.mean()
V['UTIL_QD'] = '%.1f' % (100 * util['Q-DAGFlow']); V['UTIL_BEST_BASE'] = '%.1f' % (100 * max(util[b] for b in comp))
sched = d.groupby('method').sched_time.agg(['mean', 'max'])
V['SCHED_QD_MEAN'] = '%.2f' % sched.loc['Q-DAGFlow', 'mean']; V['SCHED_QD_MAX'] = '%.1f' % sched.loc['Q-DAGFlow', 'max']
V['SCHED_BASE_MEAN'] = '%.2f' % np.mean([sched.loc[b, 'mean'] for b in comp])
# gap to strict LB
lb = df[df.method == 'LB'].set_index(['family', 'scale', 'seed']).makespan
qd = qd.copy(); qd['lb'] = [lb[(f, s, e)] for f, s, e in zip(qd.family, qd.scale, qd.seed)]
qd['gap'] = qd.makespan / qd.lb
V['GAP_MEDIAN'] = '%.2f' % qd.gap.median()
V['GAP_NONVQA_MEDIAN'] = '%.2f' % qd[qd.family != 'VQA-Loop'].gap.median()
V['GAP_NONVQA_PCT'] = '%.0f' % (100 * (qd[qd.family != 'VQA-Loop'].gap.median() - 1))
V['N_BEST_ROWS'] = str(int(sum(1 for _, r in T.iterrows() if r['Q-DAGFlow_mean'] <= min(r[m + '_mean'] for m in BASE))))
V['GAP_VQA_MEDIAN'] = '%.2f' % qd[qd.family == 'VQA-Loop'].gap.median()
qd['ub_ok'] = qd.makespan <= qd.U_plan * 1.0
V['U_COVER'] = '%.1f' % (100 * qd.ub_ok.mean())
V['TLP_LB_MEDIAN'] = '%.2f' % (qd.T_lp / qd.lb).median()
# per-baseline geometric mean ratios
for b in BASE:
    V['GM_%s' % b.replace('-', '')] = '%.3f' % summ[b]['gm_ratio']
    V['DELTA_%s' % b.replace('-', '')] = '%.2f' % summ[b]['med_delta']
V['VIOL_MAIN_QD'] = '%.3f' % qd.viol_mean.mean()
V['VIOL_MAIN_BASE'] = '%.3f' % d[d.method.isin(comp)].viol_mean.mean()
# ---- tables (markdown)
def fmt_h(x):
    return 'DNF' if not np.isfinite(x) else '%.2f' % (x / 3600)
CI = json.load(open(D + 'main_delta_ci.json'))
lines = [': **Table 5.** Mean makespan (hours) over 30 paired instances per cell; best mean per row in bold. Δ is the change of Q-DAGFlow\'s mean makespan relative to the best baseline of the row (positive = faster), with a 95% paired-bootstrap confidence interval; the marker gives the Holm-adjusted Wilcoxon result against that baseline (▲ significantly faster, ▼ slower, ○ not significant). Task counts are given in Table 4. DNF: did not finish within the 60-day horizon.', '',
         '| Family | Scale | ' + ' | '.join(METH) + ' | Δ (%) [95% CI] |', '|:------------|:-----:|' + '-------:|' * len(METH) + '----------------:|']
for _, r in T.iterrows():
    means = {m: r[m + '_mean'] for m in METH}
    best = min(means.values())
    cells = [('**%s**' % fmt_h(means[m]) if means[m] == best else fmt_h(means[m])) for m in METH]
    w = W[(W.family == r.family) & (W.scale == r.scale) & (W.base == r.best_base)].iloc[0]
    mark = '▲' if w.sig == 'better' else ('▼' if w.sig == 'worse' else '○')
    lo, hi = CI['%s|%s' % (r.family, r.scale)]
    lines.append(('| %s | %s | %s | %+.1f [%+.1f, %+.1f] %s |' % (r.family, r.scale, ' | '.join(cells), r.impr_best, lo, hi, mark)).replace('[-', '[−').replace(', -', ', −').replace('| -', '| −'))
V['TABLE_MAIN'] = '\n'.join(lines)
_r = T[(T.family == '1000Genome-H') & (T.scale == 'S')].iloc[0]
_w = W[(W.family == '1000Genome-H') & (W.scale == 'S') & (W.base == _r.best_base)].iloc[0]
V['GEN_S_BASE'] = _r.best_base; V['GEN_S_WINS'] = str(int(_w.win)); V['GEN_S_LOSSES'] = str(int(_w.loss))
V['GEN_S_MEANDIFF'] = '%.1f' % (-_r.impr_best)
lines = [': **Table 7.** Pairwise comparison of Q-DAGFlow with each baseline over the 18 family–scale settings (30 paired instances each; Holm-adjusted two-sided Wilcoxon tests, α = 0.05). GM ratio is the geometric mean of the per-instance makespan ratio Q-DAGFlow/baseline; δ is the median Cliff\'s delta (positive = Q-DAGFlow faster).', '',
         '| Baseline | Faster (sig.) | n.s. | Slower (sig.) | GM ratio | Median δ | Avg. rank |', '|:--|:-:|:-:|:-:|:-:|:-:|:-:|']
for b in BASE:
    s = summ[b]
    lines.append('| %s | %d | %d | %d | %.3f | %.2f | %.2f |' % (b, s['better'], s['ns'], s['worse'], s['gm_ratio'], s['med_delta'], ranks[b]))
lines.append('| **Q-DAGFlow** | – | – | – | 1.000 | – | **%.2f** |' % ranks['Q-DAGFlow'])
V['TABLE_STATS'] = '\n'.join(lines)
# workload table
lines = [': **Table 4.** Workload families. Task counts are medians over the 30 instances of each scale.', '',
         '| Family | Origin | Quantum task types | Qubits | S: \\|V\\| / \\|V_Q\\| | M: \\|V\\| / \\|V_Q\\| | L: \\|V\\| / \\|V_Q\\| |', '|:--|:--|:--|:--|:-:|:-:|:-:|']
meta = {'QK-HPO': ('hybrid AI (quantum-kernel model selection)', 'kernel blocks', '16–28'),
        'VQA-Loop': ('hybrid AI (variational training)', 'gradients', '16–56'),
        'SQD-Hybrid': ('quantum-centric supercomputing', 'sampling', '32–72'),
        'Montage-H': ('WfCommons Montage + quantum stages', 'K / V / S', '16–64'),
        'Epigenomics-H': ('WfCommons Epigenomics + quantum stages', 'K / V / S', '16–64'),
        '1000Genome-H': ('WfCommons 1000Genome + quantum stages', 'K / V / S', '16–64')}
for f in FAMS:
    cells = []
    for sc in 'SML':
        r = T[(T.family == f) & (T.scale == sc)].iloc[0]; cells.append('%d / %d' % (r.n_tasks, r.n_q))
    lines.append('| %s | %s | %s | %s | %s |' % (f, meta[f][0], meta[f][1], meta[f][2], ' | '.join(cells)))
V['TABLE_WORKLOADS'] = '\n'.join(lines)
for fn in ('values_rest.json', 'values_robust.json'):
    try:
        V.update(json.load(open(D + fn)))
    except FileNotFoundError:
        pass
V['SCAL_MAXN_FMT'] = '{:,}'.format(int(V['SCAL_MAXN'])) if 'SCAL_MAXN' in V else ''
os.makedirs('results', exist_ok=True); json.dump(V, open('results/values.json', 'w'), indent=1, ensure_ascii=False)
print({k: v for k, v in V.items() if not k.startswith('TABLE')})
