from figstyle import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
setup()
fig, ax = plt.subplots(figsize=(12, 5.2)); ax.set_xlim(0, 120); ax.set_ylim(0, 52); ax.axis('off')
def box(x, y, w, h, title, lines, fc='#f6f5f2', ec='#9a9993', tcol=INK, fs=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=ec, lw=0.9))
    ax.text(x + 1.3, y + h - 1.6, title, fontsize=8.4, fontweight='bold', color=tcol, va='top')
    for i, l in enumerate(lines):
        ax.text(x + 1.3, y + h - 5.0 - 2.6 * i, l, fontsize=fs, color=INK2, va='top')
def arrow(x1, y1, x2, y2, c=INK2, ls='-', rad=0.0, lw=1.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9, color=c, lw=lw, ls=ls, connectionstyle='arc3,rad=%g' % rad))
box(0.5, 27, 20.5, 23.5, 'Inputs', ['Hybrid AI workflow DAG', '• classical tasks', '  (duration, data volume)', '• quantum tasks (template,', '  qubits n, circuits N_j,', '  precision target ε)', 'QPU budget B, confidence η'])
box(0.5, 2, 20.5, 22, 'Telemetry', ['Calibration snapshots', '  (gate / readout errors)', 'Queue status per backend', 'In-job witness estimates', 'Node availability', 'Drift model (μ_d, σ_d)'])
ax.add_patch(FancyBboxPatch((25, 1.5), 64, 49, boxstyle='round,pad=0.4,rounding_size=1.5', fc='white', ec=PAL[0], lw=1.4))
ax.text(26.6, 49.6, 'Q-DAGFlow scheduler', fontsize=10, fontweight='bold', color=PAL[0], va='top')
fc, ec = '#eef4fc', '#9ec5f4'
box(31, 33.5, 56, 12.5, '①  CAMS — calibration-aware mode synthesis', ['precision-guaranteed shots per (backend, mitigation)  — Lemma 1',
    'drift-robust fidelity lower confidence bound  — Lemma 2', 'statevector emulation + queue-hedged composite modes  — Prop. 2'], fc, ec)
box(31, 18.5, 56, 12.5, '②  LP-MA — LP-certified mode assignment', ['LP: precedence + typed capacity + QPU budget  →  T_LP (bound)',
    'Markov rounding: bicriteria makespan/cost guarantee  — Theorem 1', 'budget repair and critical-path upgrade (heuristic polish)'], fc, ec)
box(31, 3.5, 56, 12.5, '③  RADAR — risk-aware dispatch & reconfiguration', ['work-conserving typed dispatch, upward rank on planned modes',
    'path-quantile risk score; type-preserving backend/hedge switching', 'witness → EWMA + CUSUM drift detection → re-plan sub-DAG'], fc, ec)
arrow(59, 33.4, 59, 31.3); arrow(59, 18.4, 59, 16.3)
arrow(30.6, 9.5, 30.6, 39.5, c=PAL[1], ls='--', rad=-0.0)
ax.text(28.0, 24.5, 'reconfigure', fontsize=7, color=PAL[1], rotation=90, va='center')
box(94, 28, 25.5, 22, 'Classical HPC tier (m nodes)', ['classical AI / scientific tasks', 'statevector emulation', '  (n ≤ 29, 16 GB/node)', 'locality-aware placement'])
box(94, 2, 25.5, 23, 'Cloud QPUs', ['Eagle-S, Eagle-K (127 q)', 'Heron-T, Heron-F (133/156 q)', 'shared heavy-tailed queues', 'billed per executed', '  QPU-second', 'queued replicas cancellable'])
arrow(21.5, 39, 30.4, 39); arrow(21.5, 13, 24.8, 13)
arrow(87.6, 11, 93.6, 38); arrow(87.6, 8, 93.6, 12)
arrow(93.6, 6, 87.8, 5.5, c=PAL[2], ls='--'); ax.text(89.0, 2.3, 'witness', fontsize=7, color=PAL[2])
save(fig, 'fig_arch')
