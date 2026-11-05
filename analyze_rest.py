"""Analysis + figures for practice/drift, ablation, budget, sensitivity, scalability and guarantee checks."""
import json, sys, numpy as np, pandas as pd
from scipy import stats
from figstyle import *
setup()
D = 'data/'
FAMS = ['QK-HPO', 'VQA-Loop', 'SQD-Hybrid', 'Montage-H', 'Epigenomics-H', '1000Genome-H']
V = {}
def load(name):
    return pd.DataFrame([r for r in json.load(open(D + name)) if 'error' not in r])

def practice():
    pr = load('exp_practice.json'); mn = load('exp_main.json')
    mn = mn[mn.scale == 'M']
    rows = []
    for f in FAMS:
        r = {'family': f}
        for b in ['HEFT-Q', 'PEFT-Q', 'QG-Dyn']:
            pub = pr[(pr.family == f) & (pr.label == b + '[pub]')]
            cam = mn[(mn.family == f) & (mn.method == b)]
            r[b + '_pub'] = pub.viol_mean.mean(); r[b + '_cams'] = cam.viol_mean.mean()
            r[b + '_ms_pub'] = pub.makespan.mean(); r[b + '_ms_cams'] = cam.makespan.mean()
        r['QD'] = mn[(mn.family == f) & (mn.method == 'Q-DAGFlow')].viol_mean.mean()
        rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(D + 'practice_table.csv', index=False)
    pub_all = np.mean([T[b + '_pub'].mean() for b in ['HEFT-Q', 'PEFT-Q', 'QG-Dyn']])
    cams_all = np.mean([T[b + '_cams'].mean() for b in ['HEFT-Q', 'PEFT-Q', 'QG-Dyn']])
    V.update(VIOL_PUB='%.3f' % pub_all, VIOL_CAMS='%.3f' % cams_all, VIOL_QD='%.3f' % T.QD.mean())
    ms_overhead = np.mean([(T[b + '_ms_cams'] / T[b + '_ms_pub']).mean() - 1 for b in ['HEFT-Q', 'PEFT-Q', 'QG-Dyn']])
    V['CAMS_MS_OVERHEAD'] = '%.1f' % (100 * ms_overhead)
    lines = [': **Table 10.** Mean per-task probability of violating the precision requirement at scale M (30 paired instances). "pub": shots planned from the latest published calibration and rescaling with published fidelities; "CAMS": the same scheduler with drift-robust shots and in-job witness rescaling.', '',
             '| Family | HEFT-Q pub | HEFT-Q CAMS | PEFT-Q pub | PEFT-Q CAMS | QG-Dyn pub | QG-Dyn CAMS | Q-DAGFlow |', '|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|']
    for _, r in T.iterrows():
        lines.append('| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | **%.3f** |' % (r.family, r.HEFT_Q_pub if False else r['HEFT-Q_pub'], r['HEFT-Q_cams'], r['PEFT-Q_pub'], r['PEFT-Q_cams'], r['QG-Dyn_pub'], r['QG-Dyn_cams'], r.QD))
    lines.append('| **Mean** | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | **%.3f** |' % tuple([T[c].mean() for c in ['HEFT-Q_pub', 'HEFT-Q_cams', 'PEFT-Q_pub', 'PEFT-Q_cams', 'QG-Dyn_pub', 'QG-Dyn_cams', 'QD']]))
    V['TABLE_PRACTICE'] = '\n'.join(lines)
    print(T.round(3).to_string())
    # drift stress
    dr = pr[pr.label.str.contains(r'\|')].copy()
    dr['meth'] = dr.label.str.split('|').str[0]; dr['ds'] = dr.label.str.split('|').str[1].astype(float)
    g = dr.groupby(['family', 'meth', 'ds'])[['viol_mean', 'makespan']].mean().reset_index()
    g.to_csv(D + 'drift_table.csv', index=False)
    fams = ['VQA-Loop', 'SQD-Hybrid', 'Montage-H']
    col = {'QG-Dyn[pub]': PAL[4], 'QG-Dyn[CAMS]': PAL[1], 'Q-DAGFlow-noW': PAL[2], 'Q-DAGFlow': PAL[0]}
    lab = {'QG-Dyn[pub]': 'QG-Dyn, published calibration', 'QG-Dyn[CAMS]': 'QG-Dyn + CAMS', 'Q-DAGFlow-noW': 'Q-DAGFlow w/o witness', 'Q-DAGFlow': 'Q-DAGFlow'}
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.0))
    for k, f in enumerate(fams):
        for m in ['QG-Dyn[pub]', 'QG-Dyn[CAMS]', 'Q-DAGFlow-noW', 'Q-DAGFlow']:
            s = g[(g.family == f) & (g.meth == m)].sort_values('ds')
            axs[0, k].plot(s.ds, s.viol_mean, marker='o', ms=3.5, color=col[m], label=lab[m])
            base = g[(g.family == f) & (g.meth == 'QG-Dyn[pub]')].sort_values('ds').makespan.values
            axs[1, k].plot(s.ds, s.makespan.values / base, marker='o', ms=3.5, color=col[m], label=lab[m])
        axs[0, k].set_title(f); axs[0, k].set_ylabel('precision-violation prob.'); axs[0, k].set_xlabel('drift intensity (× base)')
        axs[0, k].axhline(0.01, color=INK2, ls=':', lw=0.8)
        axs[1, k].set_ylabel('makespan / QG-Dyn (published)'); axs[1, k].set_xlabel('drift intensity (× base)')
        axs[1, k].axhline(1, color=INK2, ls=':', lw=0.8)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, 'fig_drift')
    for f in fams:
        a = g[(g.family == f) & (g.ds == 1.0)].set_index('meth').viol_mean
        V['DRIFT1_%s' % f.split('-')[0].upper()] = '%.3f/%.3f' % (a['QG-Dyn[pub]'], a['Q-DAGFlow'])
    a = g[g.ds == 4.0].groupby('meth').viol_mean.mean()
    V['VIOL_D4_PUB'] = '%.3f' % a['QG-Dyn[pub]']; V['VIOL_D4_QD'] = '%.3f' % a['Q-DAGFlow']; V['VIOL_D4_NOW'] = '%.3f' % a['Q-DAGFlow-noW']; V['VIOL_D4_CAMS'] = '%.3f' % a['QG-Dyn[CAMS]']
    for f in fams:
        a = g[(g.family == f) & (g.ds == 4.0)].set_index('meth').viol_mean
        k = f.split('-')[0].upper()
        V['D4_%s_QD' % k] = '%.3f' % a['Q-DAGFlow']; V['D4_%s_CAMS' % k] = '%.3f' % a['QG-Dyn[CAMS]']; V['D4_%s_PUB' % k] = '%.3f' % a['QG-Dyn[pub]']; V['D4_%s_NOW' % k] = '%.3f' % a['Q-DAGFlow-noW']
    a = g[g.ds == 0.0].groupby('meth').viol_mean.mean()
    V['VIOL_D0_PUB'] = '%.3f' % a['QG-Dyn[pub]']; V['VIOL_D0_QD'] = '%.3f' % a['Q-DAGFlow']
    # makespan cost of drift protection (ratio of means, as plotted in Fig. 7)
    rr = {}
    for m in ['Q-DAGFlow', 'QG-Dyn[CAMS]']:
        x = g[g.meth == m].set_index(['family', 'ds']).makespan
        y = g[g.meth == 'QG-Dyn[pub]'].set_index(['family', 'ds']).makespan
        rr[m] = 100 * (x / y.reindex(x.index) - 1)
    q = rr['Q-DAGFlow']
    V['DRIFT_QD_MS_D2'] = '%.0f' % max(0, q[q.index.get_level_values('ds') <= 2.0].max())
    V['DRIFT_QD_MS_VQA_D4'] = '%.0f' % q.loc[('VQA-Loop', 4.0)]
    c = rr['QG-Dyn[CAMS]'].loc['VQA-Loop']
    V['DRIFT_CAMS_MS_VQA_LO'] = '%.0f' % c.min(); V['DRIFT_CAMS_MS_VQA_HI'] = '%.0f' % c.max()
    # Table 7: CAMS-equipped methods on the scientific workflows
    wf = T[T.family.isin(['Montage-H', 'Epigenomics-H', '1000Genome-H'])].set_index('family')
    diffs = []
    for f, r in wf.iterrows():
        for b in ['HEFT-Q', 'PEFT-Q', 'QG-Dyn']:
            if f == 'Montage-H' and b != 'QG-Dyn':
                continue
            diffs.append(abs(r.QD - r[b + '_cams']))
    V['PRAC_WF_MAXDIFF'] = '%.3f' % max(diffs)
    V['PRAC_MON_HEFT'] = '%.3f' % wf.loc['Montage-H', 'HEFT-Q_cams']; V['PRAC_MON_PEFT'] = '%.3f' % wf.loc['Montage-H', 'PEFT-Q_cams']

ABL = ['w/o hedging', 'w/o LP-MA', 'w/o path-risk', 'w/o switching', 'w/o witness/CPD', 'w/o reconfig', 'w/o LCB']
def ablation():
    ab = load('exp_ablation.json')
    piv = ab.pivot_table(index=['family', 'seed'], columns='label', values='makespan', aggfunc='first')
    vio = ab.pivot_table(index=['family'], columns='label', values='viol_mean', aggfunc='mean')
    rows = []; M = np.zeros((len(FAMS), len(ABL))); P = np.ones_like(M)
    for i, f in enumerate(FAMS):
        p = piv.loc[f]
        for k, a in enumerate(ABL):
            r = p[a] / p['Full']
            M[i, k] = 100 * (np.exp(np.mean(np.log(r))) - 1)
            try:
                P[i, k] = stats.wilcoxon(p[a], p['Full']).pvalue
            except ValueError:
                P[i, k] = 1.0
            rows.append(dict(family=f, variant=a, delta_pct=M[i, k], p=P[i, k], viol=vio.loc[f, a], viol_full=vio.loc[f, 'Full']))
    T = pd.DataFrame(rows); T.to_csv(D + 'ablation_table.csv', index=False)
    fig = plt.figure(figsize=(11, 3.5))
    ax = fig.add_axes([0.10, 0.25, 0.56, 0.68]); ax2 = fig.add_axes([0.78, 0.25, 0.20, 0.68])
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f2f1ee', '#eb6834'])
    im = ax.imshow(np.clip(M, -20, 20), cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-20, vmax=20), aspect='auto')
    for i in range(M.shape[0]):
        for k in range(M.shape[1]):
            star = '*' if P[i, k] < 0.05 / M.size else ''
            ax.text(k, i, '%+.1f%s' % (M[i, k], star), ha='center', va='center', fontsize=7.2, color=INK)
    ax.set_xticks(range(len(ABL))); ax.set_xticklabels(ABL, rotation=25, ha='right'); ax.set_yticks(range(len(FAMS))); ax.set_yticklabels(FAMS)
    ax.grid(False); ax.set_title('(a) Makespan change vs. full Q-DAGFlow (%)')
    labs = ['Full'] + ABL
    vals = [vio[l].mean() for l in labs]
    cols = [METHOD_COLOR['Q-DAGFlow']] + ['#9a9993'] * len(ABL)
    ax2.barh(range(len(labs)), vals, color=cols, height=0.6)
    ax2.set_yticks(range(len(labs))); ax2.set_yticklabels(labs, fontsize=7); ax2.invert_yaxis()
    ax2.set_xlabel('mean violation probability'); ax2.set_title('(b) Precision violations')
    ax2.axvline(0.01, color=INK2, ls=':', lw=0.8)
    for y, v_ in enumerate(vals):
        ax2.text(v_ + 0.004, y, '%.3f' % v_, va='center', fontsize=6.6, color=INK2)
    ax2.set_xlim(0, max(vals) * 1.35)
    save(fig, 'fig_ablation')
    avg = M.mean(axis=0)
    for k, a in enumerate(ABL):
        V['ABL_' + a.replace('w/o ', '').replace('/', '').replace('-', '').replace(' ', '').upper()] = '%.1f' % avg[k]
    ql = ab.pivot_table(index='family', columns='label', values='qlat_p95', aggfunc='mean')
    V['ABL_HEDGE_QLATP95'] = '%.0f' % (100 * (ql.loc[['VQA-Loop', 'SQD-Hybrid', 'Montage-H'], 'w/o hedging'] / ql.loc[['VQA-Loop', 'SQD-Hybrid', 'Montage-H'], 'Full'] - 1).mean())
    V['ABL_MAX_HEDGE'] = '%.1f' % M[:, 0].max()
    V['ABL_HEDGE_VQA'] = '%.1f' % M[1, 0]; V['ABL_HEDGE_SQD'] = '%.1f' % M[2, 0]
    V['ABL_SW_VQA'] = '%.1f' % M[1, 3]; V['ABL_SW_SQD'] = '%.1f' % M[2, 3]; V['ABL_SW_MON'] = '%.1f' % M[3, 3]
    V['ABL_LP_VQA'] = '%.1f' % M[1, 1]; V['ABL_LP_VQA_ABS'] = '%.1f' % abs(M[1, 1])
    V['ABL_LP_MAXABS'] = '%.1f' % np.abs(M[:, 1]).max()
    V['ABL_WIT_VQA_ABS'] = '%.1f' % abs(M[1, ABL.index('w/o witness/CPD')])
    V['ABL_WIT_VQA_FOLD'] = '%.0f' % (vio.loc['VQA-Loop', 'w/o witness/CPD'] / max(vio.loc['VQA-Loop', 'Full'], 1e-9))
    V['ABL_LCB_ABS'] = '%.1f' % abs(M[:, ABL.index('w/o LCB')].mean())
    em = piv['w/o emulation'] / piv['Full']
    V['ABL_EMU_VQA'] = '%.1f' % (100 * (np.exp(np.mean(np.log(em.loc['VQA-Loop']))) - 1))
    V['ABL_EMU_WF_MIN'] = '%.0f' % min(np.exp(np.mean(np.log(em.loc[f]))) for f in ['Montage-H', 'Epigenomics-H', '1000Genome-H'])
    V['ABL_EMU_QK_DNF'] = '%d' % int((~np.isfinite(piv.loc['QK-HPO']['w/o emulation'])).sum())
    V['ABL_VIOL_FULL'] = '%.3f' % vio['Full'].mean(); V['ABL_VIOL_LCB'] = '%.3f' % vio['w/o LCB'].mean(); V['ABL_VIOL_WIT'] = '%.3f' % vio['w/o witness/CPD'].mean()
    V['ABL_VIOL_WIT_VQA'] = '%.3f' % vio.loc['VQA-Loop', 'w/o witness/CPD']; V['ABL_VIOL_FULL_VQA'] = '%.3f' % vio.loc['VQA-Loop', 'Full']
    print(pd.DataFrame(M, index=FAMS, columns=ABL).round(1).to_string()); print(vio.round(3).to_string())

def ablation_L():
    ab = load('exp_ablation_L.json')
    piv = ab.pivot_table(index=['family', 'seed'], columns='label', values='makespan', aggfunc='first')
    fams = ['VQA-Loop', 'SQD-Hybrid', 'Epigenomics-H', '1000Genome-H']
    cols = ['w/o hedging', 'w/o LP-MA', 'w/o switching', 'QG-Dyn']
    lines = [': **Table 11.** Ablation at scale L (10 paired instances per family): makespan change relative to full Q-DAGFlow (geometric mean, %; positive = slower). QG-Dyn is shown for reference.', '',
             '| Family | w/o hedging | w/o LP-MA | w/o switching | QG-Dyn |', '|:--|:-:|:-:|:-:|:-:|']
    for f in fams:
        p = piv.loc[f]
        vals = [100 * (np.exp(np.mean(np.log(p[c] / p['Full']))) - 1) for c in cols]
        for c, v_ in zip(cols, vals):
            V['ABLL_%s_%s' % (f.split('-')[0].upper(), c.replace('w/o ', '').replace('-', '').replace(' ', '').upper())] = '%.1f' % v_
        lines.append('| %s | %s |' % (f, ' | '.join('%+.1f' % v_ for v_ in vals)))
    V['TABLE_ABL_L'] = '\n'.join(lines)
    print('\n'.join(lines))

def budget():
    bu = load('exp_budget.json')
    fams = ['VQA-Loop', 'SQD-Hybrid', 'Epigenomics-H']
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, f in zip(axs, fams):
        s = bu[bu.family == f]
        q = s[s.label.str.startswith('QD|')].copy(); q['bf'] = q.label.str.split('|').str[1]
        g = q.groupby('bf')[['qpu_cost', 'makespan', 'budget']].mean()
        g = g.sort_values('qpu_cost')
        ax.plot(g.qpu_cost / 3600, g.makespan / 3600, marker='o', color=METHOD_COLOR['Q-DAGFlow'], label='Q-DAGFlow (budget sweep)', zorder=3)
        gg = g.copy(); gg['bfv'] = [np.inf if b_ == 'inf' else float(b_) for b_ in gg.index]; gg = gg.sort_values('bfv')
        groups = []
        for bf, row in gg.iterrows():
            if groups and abs(row.qpu_cost - groups[-1][1].qpu_cost) < 0.01 * gg.qpu_cost.max() and abs(row.makespan - groups[-1][1].makespan) < 0.01 * gg.makespan.max():
                groups[-1][0].append(bf)
            else:
                groups.append(([bf], row))
        fmt = lambda b_: '∞' if b_ == 'inf' else b_ + '×'
        for bfs, row in groups:
            lab = 'B=' + (fmt(bfs[0]) if len(bfs) == 1 else '%s…%s' % (fmt(bfs[0]), fmt(bfs[-1])))
            ax.annotate(lab, (row.qpu_cost / 3600, row.makespan / 3600), textcoords='offset points', xytext=(4, 3), fontsize=6.3, color=INK2)
        for m in ['HEFT-Q', 'PEFT-Q', 'SHEFT-Q', 'QG-Dyn', 'CF-Dyn']:
            r = s[s.label == m][['qpu_cost', 'makespan']].mean()
            ax.scatter(r.qpu_cost / 3600, r.makespan / 3600, marker=METHOD_MARK[m], color=METHOD_COLOR[m], s=30, label=m, zorder=2, edgecolor='white', linewidth=0.5)
        ax.set_title(f); ax.set_xlabel('QPU spend (billed QPU-hours)'); ax.set_ylabel('mean makespan (h)')
        # budget adherence
        adh = (q.qpu_cost <= q.budget * 1.02).mean()
        V['BUDGET_ADH_' + f.split('-')[0].upper()] = '%.0f' % (100 * adh)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, 'fig_budget')
    e_ = bu[bu.family == 'Epigenomics-H']
    cf = e_[e_.label == 'CF-Dyn'].makespan.mean(); qdm = e_[e_.label == 'QD|inf'].makespan.mean()
    V['BUDGET_EPI_CF_MS'] = '%.0f' % (100 * (cf / qdm - 1))
    vq = bu[bu.family == 'VQA-Loop']
    q2 = vq[vq.label == 'QD|2'][['makespan', 'qpu_cost']].mean(); qg = vq[vq.label == 'QG-Dyn'][['makespan', 'qpu_cost']].mean()
    V['BUD_VQA_2X_MS'] = '%.0f' % (100 * (1 - q2.makespan / qg.makespan)); V['BUD_VQA_2X_COST'] = '%.0f' % (100 * (1 - q2.qpu_cost / qg.qpu_cost))
    sq = bu[bu.family == 'SQD-Hybrid']
    qmin = sq[sq.label.str.startswith('QD|')].groupby('label')[['makespan', 'qpu_cost']].mean().qpu_cost.min()
    V['BUD_SQD_CF_COST'] = '%.1f' % (100 * (1 - sq[sq.label == 'CF-Dyn'].qpu_cost.mean() / qmin))
    # minimum feasible spend per instance (cheapest precision-feasible mode of every quantum task at t=0)
    from qdag.model import Platform, Env
    from qdag.workflows import make
    from qdag.policies import QDAGFlow
    cmin = {}
    for f in fams:
        for sd in sorted(bu[bu.family == f].seed.unique()):
            p_ = Platform(); e_ = Env(p_, int(sd)); w_ = make(f, 'M', int(sd))
            pol = QDAGFlow(w_, p_, e_, 1e15)
            ee = pol.e_plan(0.0)
            cmin[(f, sd)] = sum(min(m.cost for m in pol.menu_h(j, ee, None)) for j in range(w_.n) if w_.isq[j])
    q = bu[bu.label.str.startswith('QD|') & (bu.label != 'QD|inf')].copy()
    q['cmin'] = [cmin[(f, sd)] for f, sd in zip(q.family, q.seed)]
    feas = q[q.budget >= q.cmin]
    V['BUDGET_N_FEAS'] = '%d' % len(feas); V['BUDGET_N_ALL'] = '%d' % len(q)
    V['BUDGET_ADH_ALL'] = '%.0f' % (100 * (feas.qpu_cost <= feas.budget * 1.05).mean())
    for f in fams:
        ff = feas[feas.family == f]
        V['BUDGET_ADH5_' + f.split('-')[0].upper()] = '%.0f' % (100 * (ff.qpu_cost <= ff.budget * 1.05).mean())
    V['BUDGET_OVER_MAX'] = '%.1f' % (100 * max(0, (feas.qpu_cost / feas.budget).max() - 1))
    V['BUDGET_OVER_MED'] = '%.1f' % (100 * (feas.qpu_cost / feas.budget - 1).clip(lower=0).median())
    infe = q[q.budget < q.cmin]
    V['BUDGET_INFEAS_RATIO'] = '%.2f' % (infe.qpu_cost / infe.cmin).median() if len(infe) else 'n/a'
    print('feasible budgets', len(feas), 'of', len(q), 'adherence(5%%)', V['BUDGET_ADH_ALL'], 'max over', V['BUDGET_OVER_MAX'], 'infeasible spend/cmin', V['BUDGET_INFEAS_RATIO'])

def sens():
    se = load('exp_sens.json')
    se['meth'] = se.label.str.split('|').str[0]; se['par'] = se.label.str.split('|').str[1]; se['val'] = se.label.str.split('|').str[2].astype(float)
    fams = ['VQA-Loop', 'SQD-Hybrid', 'Epigenomics-H']
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.0))
    fc = {'VQA-Loop': PAL[0], 'SQD-Hybrid': PAL[1], 'Epigenomics-H': PAL[2]}
    for ax, par, xl in zip(axs, ['queue', 'nodes', 'sigq', 'cancel'], ['queue-wait scale (× base)', 'cluster nodes m', 'queue dispersion σ_q', 'cancellation latency (s)']):
        for f in fams:
            s = se[(se.family == f) & (se.par == par)]
            if par in ('queue', 'nodes'):
                piv = s.pivot_table(index=['seed', 'val'], columns='meth', values='makespan', aggfunc='first')
                best = piv[['HEFT-Q', 'PEFT-Q', 'QG-Dyn']].min(axis=1)
                r = (piv['Q-DAGFlow'] / best).groupby(level='val').apply(lambda x: np.exp(np.mean(np.log(x))))
                ylab = 'Q-DAGFlow / best baseline'
            elif par == 'sigq':
                piv = s.pivot_table(index=['seed', 'val'], columns='meth', values='makespan', aggfunc='first')
                r = (piv['Q-DAGFlow'] / piv['QG-Dyn']).groupby(level='val').apply(lambda x: np.exp(np.mean(np.log(x))))
                ylab = 'Q-DAGFlow / QG-Dyn'
            else:
                piv = s.pivot_table(index='seed', columns='val', values='makespan', aggfunc='first')
                ref = piv[5.0]
                r = piv.apply(lambda c: np.exp(np.mean(np.log(c / ref))))
                ylab = 'makespan / (5-s cancellation)'
            ax.plot(r.index, r.values, marker='o', ms=3.5, color=fc[f], label=f)
            k = f.split('-')[0].upper()
            if par == 'sigq':
                V['SIGQ_%s_LO' % k] = '%.2f' % r.loc[0.5]; V['SIGQ_%s_HI' % k] = '%.2f' % r.loc[1.2]
            if par == 'cancel':
                V['CANCEL120_%s' % k] = '%.1f' % (100 * (r.loc[120.0] - 1)); V['CANCEL600_%s' % k] = '%.1f' % (100 * (r.loc[600.0] - 1))
            if par == 'nodes':
                V['NODES16_%s' % k] = '%.2f' % r.loc[16.0]
            if par in ('queue', 'nodes'):
                V['SENS_%s_%s' % (par.upper(), f.split('-')[0].upper())] = '%.2f–%.2f' % (r.min(), r.max())
        ax.axhline(1, color=INK2, ls=':', lw=0.8); ax.set_xlabel(xl); ax.set_ylabel(ylab)
        if par in ('queue', 'nodes', 'cancel'):
            from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
            ax.set_xscale('log')
            tv = {'queue': [0.25, 0.5, 1, 2, 4], 'nodes': [16, 32, 64, 128, 256, 500], 'cancel': [1, 5, 30, 120, 600]}[par]
            ax.xaxis.set_major_locator(FixedLocator(tv)); ax.xaxis.set_major_formatter(FixedFormatter([('%g' % x) for x in tv]))
            ax.xaxis.set_minor_locator(NullLocator())
    axs[0].legend(loc='best')
    fig.tight_layout(); save(fig, 'fig_sens')

def scal():
    sc = load('exp_scal.json')
    g = sc.groupby(['family', 'label', 'scale'])[['n_tasks', 'sched_time', 'makespan']].mean().reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.1))
    for m in ['HEFT-Q', 'PEFT-Q', 'QG-Dyn', 'Q-DAGFlow']:
        s = g[g.label == m]
        axs[0].scatter(s.n_tasks, s.sched_time, marker=METHOD_MARK[m], color=METHOD_COLOR[m], s=22, label=m, edgecolor='white', linewidth=0.4)
    axs[0].set_xscale('log'); axs[0].set_yscale('log'); axs[0].set_xlabel('tasks |V|'); axs[0].set_ylabel('scheduler CPU time per workflow (s)')
    axs[0].set_title('(a) Scheduling overhead'); axs[0].legend()
    q = g[g.label == 'Q-DAGFlow'].copy()
    q['ratio'] = q.sched_time / (q.makespan)
    for f, c in zip(['QK-HPO', 'VQA-Loop', 'SQD-Hybrid', 'Montage-H'], PAL):
        s = q[q.family == f].sort_values('n_tasks')
        axs[1].plot(s.n_tasks, 100 * s.ratio, marker='o', ms=3.5, color=c, label=f)
    axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_xlabel('tasks |V|'); axs[1].set_ylabel('overhead / makespan (%)')
    axs[1].set_title('(b) Q-DAGFlow overhead relative to makespan'); axs[1].legend()
    fig.tight_layout(); save(fig, 'fig_scal')
    V['SCAL_MAXN'] = '%d' % int(sc.n_tasks.max())
    V['SCAL_MAX_T'] = '%.1f' % q.sched_time.max()
    V['SCAL_MAX_RATIO'] = '%.3f' % (100 * q.ratio.max())
    print(g.round(3).to_string())

if __name__ == '__main__':
    for name in sys.argv[1:]:
        globals()[name]()
    try:
        old = json.load(open(D + 'values_rest.json'))
    except FileNotFoundError:
        old = {}
    old.update(V); json.dump(old, open(D + 'values_rest.json', 'w'), indent=1)
    print(V)
