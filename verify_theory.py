"""Numerical checks of the analytical results of Sections 3-4 (independent of the scheduler code paths).

Each check draws many random instances and compares an empirical quantity with the bound proved in the paper.
Usage: python verify_theory.py   (about one minute; prints one line per check and PASS/FAIL)
"""
import math
import numpy as np
from scipy.optimize import linprog
from scipy.stats import binom, norm

rng = np.random.default_rng(20261103)
results = []


def report(name, ok, detail):
    results.append(ok)
    print('%-58s %s  %s' % (name, 'PASS' if ok else 'FAIL', detail), flush=True)


# ------------------------------------------------------------------ Lemma 1(a): raw estimation, Hoeffding + union
def lemma1a(trials=4000):
    N, eta, eps, kap = 20, 0.05, 0.08, 0.5
    worst = 0.0
    for F in (0.85, 0.9, 0.95):
        beta = (1 - F) * kap
        s = math.ceil(1.0 * math.log(2 * N / eta) / (2 * (eps - beta) ** 2))
        theta = rng.uniform(0, kap, N)
        p = F * theta + (1 - F) * 2.0 ** -8                 # kernel: all-zero probability
        est = rng.binomial(s, p, size=(trials, N)) / s       # raw estimator (no rescaling)
        fail = (np.abs(est - theta).max(axis=1) > eps).mean()
        worst = max(worst, fail)
    report('Lemma 1(a) raw estimation: P[fail] <= eta', worst <= eta, 'max empirical %.4f, eta %.2f' % (worst, eta))


# ------------------------------------------------------------------ Lemma 1(b): witness-rescaled estimation
def lemma1b(trials=3000):
    N, eta, eps, kap, xi, n = 20, 0.05, 0.08, 0.5, 0.22, 8
    pm = 2.0 ** -n
    worst = 0.0
    for F in (0.8, 0.9):
        Lw = math.log(2 * (N + 1) / eta)
        best = None
        for D in np.geomspace(1e-4, 0.5, 80) * F:
            beta = kap * (xi * F * (1 - F) + D) / (F - D)
            if eps - beta <= 0.1 * eps:
                continue
            s = math.ceil((1 / (F - D)) ** 2 * Lw / (2 * (eps - beta) ** 2))
            sw = math.ceil(Lw / (2 * (D * (1 - pm)) ** 2))
            if best is None or N * s + sw < best[0]:
                best = (N * s + sw, s, sw)
        _, s, sw = best
        theta = rng.uniform(0, kap, N)
        # adversarial residual: |r| = xi F (1-F) kappa with the sign that pushes the estimate away from theta
        r = xi * F * (1 - F) * kap * np.where(theta > kap / 2, 1, -1)
        p = np.clip(F * theta + (1 - F) * pm + r, 0.0, 1.0)
        pw = F + (1 - F) * pm
        fails = 0
        for _ in range(trials):
            Fh = (rng.binomial(sw, pw) / sw - pm) / (1 - pm)
            pt = rng.binomial(s, p) / s
            est = (pt - (1 - Fh) * pm) / Fh
            fails += np.abs(est - theta).max() > eps
        worst = max(worst, fails / trials)
    report('Lemma 1(b) witness-rescaled estimation: P[fail] <= eta', worst <= eta, 'max empirical %.4f, eta %.2f' % (worst, eta))


# ------------------------------------------------------------------ Lemma 1(c): sampling, exact binomial tail
def lemma1c():
    eta, ok, worst = 0.01, True, 0.0
    for nv in (1_000, 50_000, 100_000):
        for N in (1, 2, 10):
            for F in (0.05, 0.2, 0.6):
                psi = F + (1 - F) * 0.02
                L = math.log(N / eta)
                s = math.ceil((nv + L + math.sqrt(L * L + 2 * L * nv)) / psi)
                p_short = binom.cdf(nv - 1, s, psi)               # exact P[V < nv]
                p_task = 1 - (1 - p_short) ** N
                worst = max(worst, p_task)
                ok &= p_task <= eta
    report('Lemma 1(c) sampling: exact P[any circuit short] <= eta', ok, 'max exact %.2e, eta %.2f' % (worst, eta))


# ------------------------------------------------------------------ eq. (4): monotonicity used in Lemma 2
def beta_monotone():
    ok = True
    for xi in (0.22, 0.61):
        for D in (1e-3, 1e-2, 5e-2):
            F = np.linspace(D + 1e-3, 1, 2000)
            b = (xi * F * (1 - F) + D) / (F - D)
            ok &= bool(np.all(np.diff(b) <= 1e-12))
    report('Eq. (4): beta_mit non-increasing in F', ok, 'grid over xi, Delta')


# ------------------------------------------------------------------ Lemma 2: window-uniform drift bound
def lemma2(paths=20000, steps=600):
    mu_d, sd_d, eta_d = 0.005, 0.03, 0.05
    z = norm.ppf(1 - eta_d / 2)
    worst = 0.0
    for h in (0.5, 3.0, 12.0):
        dt = h / steps
        inc = mu_d * dt + sd_d * math.sqrt(dt) * rng.standard_normal((paths, steps))
        loge = np.cumsum(inc, axis=1)
        thr = mu_d * h + z * sd_d * math.sqrt(h)
        p = (loge.max(axis=1) > thr).mean()
        worst = max(worst, p)
    report('Lemma 2: P[multiplier exceeds bound in window] <= eta_d', worst <= eta_d, 'max empirical %.4f, eta_d %.2f' % (worst, eta_d))


# ------------------------------------------------------------------ Proposition 2: hedging
def prop2(n=400000):
    ok = True
    for mu1, mu2, o1, o2, tc in ((240, 900, 600, 300, 5), (120, 120, 1200, 1300, 30), (420, 240, 50, 2000, 60)):
        W1 = mu1 * np.exp(0.9 * rng.standard_normal(n)); W2 = mu2 * np.exp(0.9 * rng.standard_normal(n))
        TH = np.minimum(W1, W2)
        dbl = np.abs(W1 - W2) < tc
        cost = np.where(W1 <= W2, o1, o2) + dbl * np.where(W1 <= W2, o2, o1)
        bound = max(o1, o2) + min(o1, o2) * dbl.mean()
        ok &= TH.mean() <= min(W1.mean(), W2.mean()) + 1e-9
        ok &= cost.mean() <= bound + 1e-9
        x = np.quantile(TH, 0.9)
        ok &= abs((TH > x).mean() - (W1 > x).mean() * (W2 > x).mean()) < 3e-3
    report('Prop. 2: product tail, E[T_H] <= min E[T_r], cost bound', ok, '3 parameter sets')


# ------------------------------------------------------------------ Lemma 3 and Theorem 1 on random instances
def random_instance(n, K):
    """Layered DAG; classical tasks (type 0 = CPU pool) and quantum tasks with mode menus."""
    layers = rng.integers(3, 7)
    lay = np.sort(rng.integers(0, layers, n))
    pred = [[i for i in range(j) if lay[i] < lay[j] and rng.random() < 0.25] for j in range(n)]
    isq = rng.random(n) < 0.5
    p = rng.uniform(1, 10, n)
    menus = {}
    for j in range(n):
        if isq[j]:
            mm = []
            for k in range(1, K):                       # QPU types 1..K-1, one unit each
                o = rng.uniform(0.5, 5); a = rng.uniform(0, 15)
                mm.append((k, a, o, o))                 # (type, non-holding delay, occupancy, cost)
            mm.append((0, 0.0, rng.uniform(5, 40), 0.0))   # emulation on the CPU pool
            menus[j] = mm
    return pred, isq, p, menus


def solve_lp(n, pred, isq, p, menus, m, K, B):
    Q = [j for j in range(n) if isq[j]]
    off, nx = {}, 0
    for j in Q:
        off[j] = nx; nx += len(menus[j])
    iS, iT = nx, nx + n
    nv = iT + 1
    A, b = [], []

    def drow(j, row):
        if isq[j]:
            for q, (k, a, o, c) in enumerate(menus[j]):
                row[off[j] + q] += a + o
            return 0.0
        return p[j]
    for j in range(n):
        for i in pred[j]:
            row = np.zeros(nv); c0 = drow(i, row); row[iS + i] += 1; row[iS + j] -= 1
            A.append(row); b.append(-c0)
        row = np.zeros(nv); c0 = drow(j, row); row[iS + j] += 1; row[iT] -= 1
        A.append(row); b.append(-c0)
    for k in range(K):
        row = np.zeros(nv); const = sum(p[j] for j in range(n) if not isq[j]) if k == 0 else 0.0
        for j in Q:
            for q, (kk, a, o, c) in enumerate(menus[j]):
                if kk == k:
                    row[off[j] + q] += o
        row[iT] = -(m if k == 0 else 1)
        A.append(row); b.append(-const)
    row = np.zeros(nv)
    for j in Q:
        for q, (k, a, o, c) in enumerate(menus[j]):
            row[off[j] + q] = c
    A.append(row); b.append(B)
    Aeq = np.zeros((len(Q), nv)); beq = np.ones(len(Q))
    for r, j in enumerate(Q):
        Aeq[r, off[j]: off[j] + len(menus[j])] = 1
    cvec = np.zeros(nv); cvec[iT] = 1
    res = linprog(cvec, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq if Q else None, b_eq=beq if Q else None,
                  bounds=[(0, 1)] * nx + [(0, None)] * (n + 1), method='highs')
    return res, off


def markov(menus, x, off, m, rho1, rho2):
    out = {}
    for j, mm in menus.items():
        xj = np.clip(x[off[j]: off[j] + len(mm)], 0, 1)
        L = np.array([a + o for (k, a, o, c) in mm]); ld = np.array([o / (m if k == 0 else 1) for (k, a, o, c) in mm])
        cost = np.array([c for (k, a, o, c) in mm])
        okm = (L <= xj @ L / rho1 + 1e-9) & (ld <= xj @ ld / rho2 + 1e-9)
        idx = np.where(okm)[0]
        out[j] = int(idx[np.argmin(cost[idx])])
    return out


def dispatch(n, pred, isq, p, menus, assign, m, K, scale):
    """Event-driven, type-wise work-conserving dispatch with realised values scale*planning (scale <= 1)."""
    import heapq
    a_r, o_r, typ = np.zeros(n), np.zeros(n), np.zeros(n, int)
    for j in range(n):
        if isq[j]:
            k, a, o, c = menus[j][assign[j]]
            a_r[j], o_r[j], typ[j] = a * scale[j, 0], o * scale[j, 1], k
        else:
            o_r[j], typ[j] = p[j] * scale[j, 1], 0
    cap = [m] + [1] * (K - 1)
    free = list(cap); waiting = [[] for _ in range(K)]
    npred = [len(pr) for pr in pred]; succ = [[] for _ in range(n)]
    for j in range(n):
        for i in pred[j]:
            succ[i].append(j)
    ev = []; fin = np.zeros(n)
    for j in range(n):
        if npred[j] == 0:
            heapq.heappush(ev, (a_r[j], 0, j))
    while ev:
        t, kind, j = heapq.heappop(ev)
        if kind == 0:                                   # delay over: request a unit of its type
            k = typ[j]
            if free[k] > 0:
                free[k] -= 1; heapq.heappush(ev, (t + o_r[j], 1, j))
            else:
                waiting[k].append(j)
        else:                                           # completion: release unit, start a waiting task
            fin[j] = t; k = typ[j]; free[k] += 1
            if waiting[k]:
                w = waiting[k].pop(0); free[k] -= 1; heapq.heappush(ev, (t + o_r[w], 1, w))
            for s_ in succ[j]:
                npred[s_] -= 1
                if npred[s_] == 0:
                    heapq.heappush(ev, (t + a_r[s_], 0, s_))
    lat = a_r + o_r
    lp = np.zeros(n)
    for j in range(n):
        lp[j] = lat[j] + max([lp[i] for i in pred[j]], default=0.0)
    W = np.zeros(K)
    for j in range(n):
        W[typ[j]] += o_r[j]
    lemma3 = lp.max() + sum(W[k] / cap[k] for k in range(K))
    return fin.max(), lemma3


def theorem1(instances=150):
    rho1 = rho2 = 1 / 3
    ok_l3 = ok_cost = ok_t1 = True
    worst_ratio = 0.0
    for _ in range(instances):
        n, K, m = int(rng.integers(15, 45)), int(rng.integers(2, 5)), int(rng.integers(2, 6))
        pred, isq, p, menus = random_instance(n, K)
        if not menus:
            continue
        cmin = sum(min(c for (_, _, _, c) in mm) for mm in menus.values())
        cmax = sum(max(c for (_, _, _, c) in mm) for mm in menus.values())
        B = cmin + rng.uniform(0.1, 0.9) * (cmax - cmin)
        res, off = solve_lp(n, pred, isq, p, menus, m, K, B)
        if res.status != 0:
            continue
        T = res.x[-1]
        a = markov(menus, res.x, off, m, rho1, rho2)
        cost = sum(menus[j][a[j]][3] for j in menus)
        ok_cost &= cost <= B / (1 - rho1 - rho2) + 1e-6
        for _ in range(5):
            scale = rng.uniform(0.3, 1.0, (n, 2))
            C, l3 = dispatch(n, pred, isq, p, menus, a, m, K, scale)
            ok_l3 &= C <= l3 + 1e-6
            ok_t1 &= C <= (1 / rho1 + K / rho2) * T + 1e-6
            worst_ratio = max(worst_ratio, C / T)
    report('Lemma 3: realised C_max <= typed list-scheduling bound', ok_l3, '750 dispatches')
    report('Theorem 1: Markov-rounded cost <= B/(1-rho1-rho2)', ok_cost, '%d instances' % instances)
    report('Theorem 1: C_max <= (1/rho1 + |K|/rho2) T*_LP', ok_t1, 'worst C_max/T*_LP = %.2f (bound >= 9)' % worst_ratio)


# ------------------------------------------------------------------ eq. (13): path-quantile score
def score_check():
    ok = True
    for _ in range(2000):
        Vr, mu_r = rng.uniform(0, 1e7), rng.uniform(0, 1e5)
        L = rng.uniform(0, 1e4, 5); s2 = rng.uniform(0, 1e7, 5); z = 1.2816
        q = mu_r + L + z * np.sqrt(Vr + s2)
        S = L + z * (np.sqrt(Vr + s2) - np.sqrt(Vr))
        ok &= int(np.argmin(q)) == int(np.argmin(S))
    report('Eq. (13): score has the same minimiser as the path quantile', ok, '2000 random menus')


if __name__ == '__main__':
    lemma1a(); lemma1b(); lemma1c(); beta_monotone(); lemma2(); prop2(); theorem1(); score_check()
    print('all checks passed' if all(results) else 'SOME CHECKS FAILED')
