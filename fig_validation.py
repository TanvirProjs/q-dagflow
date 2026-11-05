import json, numpy as np
from figstyle import *
setup()
D = '/home/claude/qdag/data/'
v = json.load(open(D + 'esp_validation.json'))
esp = json.load(open(D + 'esp_table.json'))
tim = json.load(open(D + 'emulation_timing.json'))
fig, ax = plt.subplots(1, 4, figsize=(11.2, 2.75))
# (a) predicted vs simulated
a = ax[0]
for b, c in [('Eagle-S', BACK_COLOR['Eagle-S']), ('Heron-T', BACK_COLOR['Heron-T']), ('Heron-F', BACK_COLOR['Heron-F'])]:
    x = [d['pred'] for d in v if d['backend'] == b]; y = [d['p0'] for d in v if d['backend'] == b]
    a.scatter(x, y, s=12, color=c, label=b, edgecolor='white', linewidth=0.4)
a.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls='--')
P = np.array([[d['pred'], d['p0']] for d in v]); r2 = 1 - ((P[:, 1] - P[:, 0]) ** 2).sum() / ((P[:, 1] - P[:, 1].mean()) ** 2).sum()
a.set_xlabel('ESP-model prediction  $F k+(1-F)2^{-n}$'); a.set_ylabel('Noisy simulation  $\\Pr[0^n]$')
a.set_title('(a) Bias model vs. device noise'); a.text(0.05, 0.86, '$R^2$ = %.3f\nn = %d circuits' % (r2, len(v)), transform=a.transAxes, fontsize=7.5)
a.legend(loc='lower right', handletextpad=0.2)
# (b) raw vs mitigated error
a = ax[1]
k = np.array([d['k'] for d in v]); p0 = np.array([d['p0'] for d in v]); kh = np.array([d['khat'] for d in v]); n = np.array([d['n'] for d in v])
ns = sorted(set(n))
raw = [np.mean(np.abs(p0[n == q] - k[n == q])) for q in ns]; mit = [np.mean(np.abs(kh[n == q] - k[n == q])) for q in ns]
a.plot(ns, raw, marker='o', color=PAL[1], label='raw estimate')
a.plot(ns, mit, marker='o', color=PAL[0], label='depolarising-rescaled')
a.set_xlabel('qubits $n$'); a.set_ylabel('mean |error| of kernel entry'); a.set_title('(b) Effect of rescaling'); a.legend()
# (c) ESP vs n by backend (vqa template light cone excluded: use sqd template)
a = ax[2]
for b in ['Eagle-S', 'Eagle-K', 'Heron-T', 'Heron-F']:
    rows = sorted([r for r in esp[b]['rows'] if r['template'] == 'kernel'], key=lambda r: r['n'])
    a.plot([r['n'] for r in rows], [r['esp'] for r in rows], marker='o', ms=3, color=BACK_COLOR[b], label=b)
a.set_yscale('log'); a.set_ylim(1e-3, 1.2); a.set_xlabel('qubits $n$'); a.set_ylabel('ESP (kernel circuit)')
a.set_title('(c) Calibration-derived fidelity'); a.legend(ncol=1)
# (d) emulation timing
a = ax[3]
for tmpl, c in [('kernel', PAL[0]), ('vqa', PAL[1]), ('sqd', PAL[2])]:
    rows = sorted([r for r in tim if r['template'] == tmpl], key=lambda r: r['n'])
    nn = np.array([r['n'] for r in rows]); tt = np.array([r['t_sv'] for r in rows])
    a.plot(nn, tt, marker='o', ms=3, lw=0, color=c, label=tmpl + ' (measured)')
    big = nn >= 20; f = np.polyfit(nn[big], np.log2(tt[big]), 1)
    xx = np.arange(20, 30); a.plot(xx, 2 ** np.polyval(f, xx), color=c, lw=1.0, ls='--')
a.set_yscale('log'); a.set_xlabel('qubits $n$'); a.set_ylabel('statevector time (s)'); a.set_title('(d) Classical emulation cost')
a.axvline(29, color=INK2, lw=0.8, ls=':'); a.text(28.6, 3e-3, '16 GB limit', rotation=90, fontsize=6.5, color=INK2, ha='right')
a.legend(loc='upper left')
fig.tight_layout(w_pad=1.2)
save(fig, 'fig_validation')
print('R2', r2)
