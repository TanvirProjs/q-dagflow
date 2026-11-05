import networkx as nx, numpy as np
from figstyle import *
from qdag.workflows import make
setup()
fams = ['QK-HPO', 'VQA-Loop', 'SQD-Hybrid', 'Montage-H', 'Epigenomics-H', '1000Genome-H']
fig, axs = plt.subplots(2, 3, figsize=(11, 5.6))
for ax, f in zip(axs.flat, fams):
    w = make(f, 'XS', 3)
    G = nx.DiGraph(); G.add_nodes_from(range(w.n))
    for (i, j) in w.ebytes: G.add_edge(i, j)
    gens = list(nx.topological_generations(G))
    pos = {}
    for d, layer in enumerate(gens):
        L = sorted(layer)
        for k, v in enumerate(L):
            pos[v] = ((k - (len(L) - 1) / 2) / max(1, len(L)), -d)
    eb = np.array([w.ebytes[e] for e in G.edges])
    lw = 0.3 + 1.6 * (np.log10(eb + 1) - 3) / 7
    nx.draw_networkx_edges(G, pos, ax=ax, width=np.clip(lw, 0.25, 1.8), edge_color='#b9b8b2', arrows=False)
    qn = [v for v in G if w.isq[v]]; cn = [v for v in G if not w.isq[v]]
    nx.draw_networkx_nodes(G, pos, nodelist=cn, ax=ax, node_size=14, node_color='#d6d5d0', edgecolors='#7d7c77', linewidths=0.4)
    cls_c = {'K': PAL[0], 'V': PAL[1], 'S': PAL[2]}
    nx.draw_networkx_nodes(G, pos, nodelist=qn, ax=ax, node_size=20, node_color=[cls_c[w.q[v]['cls']] for v in qn], edgecolors='white', linewidths=0.4)
    ax.set_title('%s  (%d tasks, %d quantum)' % (f, w.n, len(qn)), fontsize=8.5)
    ax.axis('off')
from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='', color='#d6d5d0', mec='#7d7c77', label='classical task'),
     Line2D([], [], marker='o', ls='', color=PAL[0], label='quantum kernel block (K)'),
     Line2D([], [], marker='o', ls='', color=PAL[1], label='variational gradient (V)'),
     Line2D([], [], marker='o', ls='', color=PAL[2], label='SQD sampling (S)')]
fig.legend(handles=h, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, 'fig_dags')
