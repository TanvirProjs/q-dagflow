import json, numpy as np, pandas as pd
from figstyle import *
setup()
FAMS = ['QK-HPO', 'VQA-Loop', 'SQD-Hybrid', 'Montage-H', 'Epigenomics-H', '1000Genome-H']
METH = ['HEFT-Q', 'PEFT-Q', 'SHEFT-Q', 'QG-Dyn', 'CF-Dyn', 'Q-DAGFlow']
df = pd.DataFrame(json.load(open('data/exp_main.json')))
lb = df[df.method == 'LB'].set_index(['family', 'scale', 'seed']).makespan
d = df[df.method != 'LB'].copy()
d['norm'] = d.makespan.values / np.array([lb[(f, s, e)] for f, s, e in zip(d.family, d.scale, d.seed)])
fig, axs = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True)
off = np.linspace(-0.3, 0.3, len(METH))
for ax, f in zip(axs.flat, FAMS):
    for k, m in enumerate(METH):
        xs, ys, lo, hi = [], [], [], []
        for i, sc in enumerate(['S', 'M', 'L']):
            v = d[(d.family == f) & (d.scale == sc) & (d.method == m)].norm.values
            v = v[np.isfinite(v)]
            mu = v.mean(); se = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
            xs.append(i + off[k]); ys.append(mu); lo.append(se); hi.append(se)
        lw = 1.8 if m == 'Q-DAGFlow' else 1.0
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=METHOD_MARK[m], color=METHOD_COLOR[m], ms=4.5 if m != 'Q-DAGFlow' else 6,
                    capsize=1.5, elinewidth=0.8, lw=0, label=m, mec='white', mew=0.5, zorder=3 if m == 'Q-DAGFlow' else 2)
    ax.set_title(f); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['S', 'M', 'L'])
    ax.set_ylabel('makespan / LB')
    ax.axhline(1, color=INK2, lw=0.6, ls=':')
for ax in axs[1]:
    ax.set_xlabel('workflow scale')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, 'fig_norm_makespan')

# ---------------- critical difference diagram
st = json.load(open('data/main_stats.json'))
ranks = st['ranks']; CD = st['CD']
order = sorted(ranks, key=ranks.get)
fig, ax = plt.subplots(figsize=(7.2, 2.1))
lo_, hi_ = 1, 7
ax.set_xlim(lo_ - 0.2, hi_ + 0.2); ax.set_ylim(0, 1); ax.axis('off')
ax.plot([lo_, hi_], [0.78, 0.78], color=INK, lw=1)
for r in range(1, 8):
    ax.plot([r, r], [0.78, 0.82], color=INK, lw=1); ax.text(r, 0.86, str(r), ha='center', fontsize=8)
for i, m in enumerate(order):
    rk = ranks[m]; left = i < 4
    y = 0.62 - 0.14 * (i if left else i - 4)
    ax.plot([rk, rk], [0.78, y], color=METHOD_COLOR[m], lw=1.2)
    xt = lo_ - 0.15 if left else hi_ + 0.15
    ax.plot([rk, xt], [y, y], color=METHOD_COLOR[m], lw=1.2)
    ax.text(xt + (-0.05 if left else 0.05), y, '%s (%.2f)' % (m, rk), ha='right' if left else 'left', va='center', fontsize=8,
            fontweight='bold' if m == 'Q-DAGFlow' else 'normal')
ax.plot([lo_, lo_ + CD], [0.97, 0.97], color=INK, lw=2); ax.text(lo_ + CD / 2, 1.0, 'CD = %.2f' % CD, ha='center', fontsize=7.5, va='bottom')
# cliques: maximal groups whose rank range < CD (sub-cliques suppressed)
srt = [ranks[m] for m in order]
cl = []
for i in range(len(order)):
    j = i
    while j + 1 < len(order) and srt[j + 1] - srt[i] < CD:
        j += 1
    if j > i and not any(a <= i and j <= b for a, b in cl):
        cl.append((i, j))
yb = 0.735
for i, j in cl:
    ax.plot([srt[i] - 0.03, srt[j] + 0.03], [yb, yb], color=INK2, lw=2.5, solid_capstyle='round'); yb -= 0.04
save(fig, 'fig_cd')
print(order, [round(ranks[m], 2) for m in order])
