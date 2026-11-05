"""LP-guided mode assignment (LP-MA): relaxation, Markov rounding, budget repair, CPM utilities."""
import math
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def cpm(wf, R, d, comm, release=None, inR=None):
    """Critical-path analysis restricted to task set R (list in topological order).
    d: dict/array of durations; comm(i,j) edge delay. Returns (EFT dict, CP length, slack dict)."""
    if inR is None:
        inR = set(R)
    est, eft = {}, {}
    for j in R:
        s = release.get(j, 0.0) if release else 0.0
        for i in wf.pred[j]:
            if i in inR:
                v = eft[i] + comm(i, j)
                if v > s:
                    s = v
        est[j] = s; eft[j] = s + d[j]
    CP = max(eft.values()) if eft else 0.0
    lft = {}
    for j in reversed(R):
        lf = CP
        for s_ in wf.succ[j]:
            if s_ in inR:
                v = lft[s_] - d[s_] - comm(j, s_)
                if v < lf:
                    lf = v
        lft[j] = lf
    slack = {j: lft[j] - eft[j] for j in R}
    return eft, CP, slack


def graham_bound(wf, R, d, occ_by, comm, m_nodes, release=None):
    """U = CP + sum_k W_k / m_k  (typed list-scheduling bound); occ_by[j] = {type: occupancy}."""
    _, CP, _ = cpm(wf, R, d, comm, release)
    W = {}
    for j in R:
        for k, o in occ_by[j].items():
            W[k] = W.get(k, 0.0) + o
    return CP + sum(w / (m_nodes if k == 'N' else 1.0) for k, w in W.items())


def path_variance(wf, R, d, v, comm, release=None):
    """For each task, variance accumulated along the longest (mean) path through it, excluding itself."""
    inR = set(R)
    eft, vf = {}, {}
    for j in R:
        s, vs = (release.get(j, 0.0) if release else 0.0), 0.0
        for i in wf.pred[j]:
            if i in inR:
                x = eft[i] + comm(i, j)
                if x > s:
                    s, vs = x, vf[i]
        eft[j] = s + d[j]; vf[j] = vs + v[j]
    tail, vt = {}, {}
    for j in reversed(R):
        bt, bv = 0.0, 0.0
        for s_ in wf.succ[j]:
            if s_ in inR:
                x = comm(j, s_) + d[s_] + tail[s_]
                if x > bt:
                    bt, bv = x, v[s_] + vt[s_]
        tail[j] = bt; vt[j] = bv
    return {j: vf[j] - v[j] + vt[j] for j in R}


def solve_mode_lp(wf, R, menus, lat, dur, comm, m_nodes, budget, release=None, use_comm=True):
    """Solve the LP relaxation of precision-constrained mode assignment on task set R.

    menus[j]: list of modes for quantum tasks; lat[j][k]: planning latency of mode k;
    occupancy/cost read from mode objects. Returns (x dict j->array, T*, status)."""
    inR = set(R)
    pos = {j: k for k, j in enumerate(R)}
    Q = [j for j in R if wf.isq[j]]
    xoff, nx = {}, 0
    for j in Q:
        xoff[j] = nx; nx += len(menus[j])
    nS = len(R)
    iS = nx; iT = nx + nS; nv = iT + 1
    rows, cols, vals, bub = [], [], [], []
    r = 0

    def add_d(j, coef):
        nonlocal rows, cols, vals
        if wf.isq[j]:
            for k, L in enumerate(lat[j]):
                rows.append(r); cols.append(xoff[j] + k); vals.append(coef * L)
            return 0.0
        return coef * dur[j]

    # precedence: S_i + d_i + c_ij - S_j <= 0
    for i in R:
        for j in wf.succ[i]:
            if j not in inR:
                continue
            const = add_d(i, 1.0)
            rows += [r, r]; cols += [iS + pos[i], iS + pos[j]]; vals += [1.0, -1.0]
            bub.append(-const - (comm(i, j) if use_comm else 0.0)); r += 1
    # makespan: S_j + d_j - T <= 0 for local sinks
    for j in R:
        if any(s in inR for s in wf.succ[j]):
            continue
        const = add_d(j, 1.0)
        rows += [r, r]; cols += [iS + pos[j], iT]; vals += [1.0, -1.0]
        bub.append(-const); r += 1
    # typed work: CPU pool and each QPU
    Wc = sum(dur[j] for j in R if not wf.isq[j])
    types = {}
    for j in Q:
        for k, md in enumerate(menus[j]):
            for key, o in md.occ_by.items():
                types.setdefault(key, []).append((xoff[j] + k, o))
    for key in sorted(set(list(types.keys()) + ['N']), key=str):   # deterministic row order (hash-seed independent)
        for c, o in types.get(key, []):
            rows.append(r); cols.append(c); vals.append(o)
        rows.append(r); cols.append(iT); vals.append(-(m_nodes if key == 'N' else 1.0))
        bub.append(-Wc if key == 'N' else 0.0); r += 1
    # budget
    if budget is not None:
        for j in Q:
            for k, md in enumerate(menus[j]):
                if md.cost > 0:
                    rows.append(r); cols.append(xoff[j] + k); vals.append(md.cost)
        bub.append(budget); r += 1
    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    # assignment equalities
    er, ec, ev = [], [], []
    for q, j in enumerate(Q):
        for k in range(len(menus[j])):
            er.append(q); ec.append(xoff[j] + k); ev.append(1.0)
    A_eq = sp.csr_matrix((ev, (er, ec)), shape=(len(Q), nv)) if Q else None
    b_eq = np.ones(len(Q)) if Q else None
    bounds = [(0, 1)] * nx + [((release.get(j, 0.0) if release else 0.0), None) for j in R] + [(0, None)]
    c = np.zeros(nv); c[iT] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.array(bub), A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.status != 0:
        return None, None, res.status
    x = {j: np.clip(res.x[xoff[j]: xoff[j] + len(menus[j])], 0, 1) for j in Q}
    return x, float(res.x[iT]), 0


def markov_round(x, lat, menus, m_nodes, rho1=1 / 3, rho2=1 / 3):
    """Deterministic Markov rounding (Theorem 2): per task keep modes with latency <= pbar/rho1 and
    normalised load <= lbar/rho2, then take the cheapest."""
    out = {}
    for j, xj in x.items():
        L = np.array(lat[j]); md = menus[j]
        ld = np.array([sum(o / (m_nodes if k == 'N' else 1.0) for k, o in m.occ_by.items()) for m in md])
        cost = np.array([m.cost for m in md])
        pbar, lbar = float(xj @ L), float(xj @ ld)
        ok = (L <= pbar / rho1 + 1e-9) & (ld <= lbar / rho2 + 1e-9)
        if not ok.any():
            ok = xj > 0
        idx = np.where(ok)[0]
        out[j] = int(idx[np.argmin(cost[idx] + 1e-9 * L[idx])])
    return out


def argmax_round(x):
    return {j: int(np.argmax(xj)) for j, xj in x.items()}


def repair_budget(wf, R, assign, menus, lat, dur, comm, budget, release=None, max_rounds=200):
    """Greedy downgrade until cost <= budget, preferring changes absorbed by slack (batched)."""
    a = dict(assign)
    tot = sum(menus[j][k].cost for j, k in a.items())
    for _ in range(max_rounds):
        if tot <= budget + 1e-6:
            break
        d = {j: (lat[j][a[j]] if wf.isq[j] else dur[j]) for j in R}
        _, _, slack = cpm(wf, R, d, comm, release)
        cands = []
        for j, k in a.items():
            cj = menus[j][k].cost
            bestj = None
            for k2, m2 in enumerate(menus[j]):
                dc = cj - m2.cost
                if dc <= 1e-9:
                    continue
                dl = lat[j][k2] - lat[j][k]
                pen = (max(0.0, dl - slack[j]) + 1e-3 * max(0.0, dl)) / dc
                if bestj is None or pen < bestj[0]:
                    bestj = (pen, j, k2, dc)
            if bestj:
                cands.append(bestj)
        if not cands:
            break
        cands.sort()
        batch = max(1, len(cands) // 20)
        for pen, j, k2, dc in cands[:batch]:
            a[j] = k2; tot -= dc
            if tot <= budget + 1e-6:
                break
    return a, tot


def upgrade_critical(wf, R, assign, menus, lat, dur, comm, budget, release=None, rounds=40):
    """Spend leftover budget on faster modes for critical tasks."""
    a = dict(assign)
    tot = sum(menus[j][k].cost for j, k in a.items())
    for _ in range(rounds):
        d = {j: (lat[j][a[j]] if wf.isq[j] else dur[j]) for j in R}
        _, CP, slack = cpm(wf, R, d, comm, release)
        best = None
        for j, k in a.items():
            if slack[j] > 1e-6 * max(1.0, CP):
                continue
            for k2, m2 in enumerate(menus[j]):
                gain = lat[j][k] - lat[j][k2]
                extra = m2.cost - menus[j][k].cost
                if gain > 1e-6 and tot + extra <= budget + 1e-6:
                    score = gain / (1.0 + max(0.0, extra))
                    if best is None or score > best[0]:
                        best = (score, j, k2, extra)
        if best is None:
            break
        _, j, k2, extra = best
        a[j] = k2; tot += extra
    return a, tot
