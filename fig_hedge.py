import numpy as np
from figstyle import *
from qdag.model import Platform, synthesize_modes, lat_stats, BACKENDS
from qdag.workflows import qtask
setup()
p = Platform()
task = qtask(None, 'V', 40)
e = {b: 1.0 for b in BACKENDS}
mu = dict(p.mu0)
mm = synthesize_modes(p, task, e, witness=True, emulation=False, hedge_mu=mu)
singles = {m.b: m for m in mm if m.kind == 'Q'}
rng = np.random.default_rng(7); N = 200000
Z = {b: rng.standard_normal(N) for b in BACKENDS}
def single(b):
    m = singles[b]; st = p.wan_up + mu[b] * np.exp(p.sig_q * Z[b]); return st + m.occ + p.wan_dn, np.full(N, m.occ)
def hedged(b1, b2):
    s1 = p.wan_up + mu[b1] * np.exp(p.sig_q * Z[b1]); s2 = p.wan_up + mu[b2] * np.exp(p.sig_q * Z[b2])
    o1, o2 = singles[b1].occ, singles[b2].occ
    w = s1 <= s2; fin = np.where(w, s1 + o1, s2 + o2) + p.wan_dn
    dbl = np.abs(s1 - s2) < 5.0
    cost = np.where(w, o1, o2) + dbl * np.where(w, o2, o1)
    return fin, cost
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.0), gridspec_kw={'width_ratios': [1.35, 1]})
a = ax[0]
rows = []
for b in BACKENDS:
    f, c = single(b); xs = np.sort(f); a.plot(xs / 60, 1 - np.arange(N) / N, color=BACK_COLOR[b], lw=1.4, label='%s only' % b)
    rows.append((b, f.mean(), np.percentile(f, 95), c.mean()))
best_pair = min([(b1, b2) for i, b1 in enumerate(BACKENDS) for b2 in BACKENDS[i + 1:]], key=lambda pr: hedged(*pr)[0].mean())
f, c = hedged(*best_pair); xs = np.sort(f)
a.plot(xs / 60, 1 - np.arange(N) / N, color=INK, lw=2.0, ls='--', label='hedged {%s, %s}' % best_pair)
rows.append(('hedged', f.mean(), np.percentile(f, 95), c.mean()))
a.set_yscale('log'); a.set_ylim(1e-3, 1); a.set_xlim(0, 160)
a.set_xlabel('completion latency (min)'); a.set_ylabel('Pr[latency > x]'); a.set_title('(a) Latency tail of one 40-qubit gradient job')
a.legend(loc='upper right')
a = ax[1]
labs = [r[0] for r in rows]; means = [r[1] / 60 for r in rows]; p95 = [r[2] / 60 for r in rows]; cost = [r[3] / 60 for r in rows]
y = np.arange(len(labs))
cols = [BACK_COLOR.get(l, INK) for l in labs]
a.barh(y - 0.18, means, height=0.34, color=cols, label='mean')
a.barh(y + 0.18, p95, height=0.34, color=cols, alpha=0.45, label='95th pct.')
for i in range(len(labs)):
    a.text(p95[i] + 1.5, y[i] + 0.18, 'cost %.1f QPU-min' % cost[i], va='center', fontsize=6.8, color=INK2)
a.set_yticks(y); a.set_yticklabels(labs); a.invert_yaxis(); a.set_xlabel('latency (min)'); a.set_title('(b) Mean / 95th pct. latency and expected cost')
a.set_xlim(0, max(p95) * 1.6)
from matplotlib.patches import Patch
a.legend(handles=[Patch(color='#6b6a66', label='mean'), Patch(color='#6b6a66', alpha=0.45, label='95th percentile')], loc='lower right')
fig.tight_layout()
save(fig, 'fig_hedge')
for r in rows: print('%-8s mean=%.1f min p95=%.1f min cost=%.1f QPU-min' % (r[0], r[1] / 60, r[2] / 60, r[3] / 60))
