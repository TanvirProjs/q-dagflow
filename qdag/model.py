"""Platform, environment and quantum-task execution-mode models for Q-DAGFlow.

All quantum-hardware parameters are derived from IBM calibration snapshots shipped with
qiskit-ibm-runtime (fake_sherbrooke, fake_kyiv, fake_torino, fake_fez); classical emulation
costs are fitted to statevector timings measured on the experiment host.
"""
import json, math, os
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
BACKENDS = ['Eagle-S', 'Eagle-K', 'Heron-T', 'Heron-F']
TEMPLATE = {'K': 'kernel', 'V': 'vqa', 'S': 'sqd'}


class Platform:
    def __init__(self, m_nodes=128, **kw):
        self.m = m_nodes
        self.mem = 16e9                  # bytes per simulated node (16 GB; modelled desktop-class node)
        self.bw = 940e6 / 8              # modelled inter-node bandwidth (B/s), typical switched Gigabit Ethernet
        self.lat = 0.18e-3               # modelled round-trip latency (s)
        self.wan_up, self.wan_dn = 1.0, 0.5   # cloud submission / retrieval latency (s)
        self.t_job, self.t_circ, self.rep = 8.0, 0.05, 250e-6   # per-job, per-circuit overhead; rep delay (snapshot default)
        self.mu0 = {'Eagle-S': 240.0, 'Eagle-K': 120.0, 'Heron-T': 420.0, 'Heron-F': 900.0}  # median queue wait (s)
        self.sig_q = 0.9
        # residual mismatch of the rescaled estimator: envelope (max, rounded up) of |k_hat-k|/(kappa*(1-F)) over the
        # density-matrix validation circuits with k <= kappa, kappa in {0.5, 1} (data/esp_validation.json)
        self.xi = {'Eagle-S': 0.61, 'Eagle-K': 0.61, 'Heron-T': 0.22, 'Heron-F': 0.22}
        self.eta = 0.01                  # precision confidence parameter (sampling error incl. witness)
        self.eta_d = 0.05                # drift confidence parameter (Lemma 2)
        self.eta_q = 0.01                # queue-tail parameter: execution must end within the planning window
        self.mu_d, self.sd_d = 0.005, 0.03   # drift model assumed by the planner (per h, per sqrt(h))
        self.drift_margin = 900.0        # s added to the planning window (upload, execution, download)
        self.r0 = 0.02                   # random post-selection pass rate for SQD sampling
        self.f_floor = 0.02              # minimum ESP for SQD sampling to be admissible
        self.max_shots = 2_000_000
        self.witness_frac = 0.02
        self.cancel_lat = 5.0
        self.queue_scale = 1.0
        for k, v in kw.items():
            setattr(self, k, v)
        self._load_tables()

    # ---------------------------------------------------------------- calibration tables
    def _load_tables(self):
        esp = json.load(open(os.path.join(DATA, 'esp_table.json')))
        self.esp_tab, self.dur_tab = {}, {}
        for b in BACKENDS:
            for tmpl in ('kernel', 'vqa', 'sqd'):
                rows = sorted([r for r in esp[b]['rows'] if r['template'] == tmpl], key=lambda r: r['n'])
                ns = np.array([r['n'] for r in rows], float)
                le = np.log(np.maximum(1e-6, np.array([r['esp'] for r in rows])))
                du = np.array([r['duration_s'] for r in rows])
                self.esp_tab[(b, tmpl)] = (ns, le)
                self.dur_tab[(b, tmpl)] = (ns, du)
        tim = json.load(open(os.path.join(DATA, 'emulation_timing.json')))
        self.sv_fit = {}
        for tmpl in ('kernel', 'vqa', 'sqd'):
            rows = sorted([r for r in tim if r['template'] == tmpl], key=lambda r: r['n'])
            ns = np.array([r['n'] for r in rows], float)
            tsv = np.array([r['t_sv'] for r in rows]); tip = np.array([r['t_ip'] for r in rows])
            big = ns >= 20
            a = np.polyfit(ns[big], np.log2(tsv[big]), 1)
            b = np.polyfit(ns[big], np.log2(np.maximum(tip[big], 1e-7)), 1)
            self.sv_fit[tmpl] = (ns, tsv, tip, a, b)

    def esp0(self, b, tmpl, n):
        ns, le = self.esp_tab[(b, tmpl)]
        if n <= ns[-1]:
            return float(np.exp(np.interp(n, ns, le)))
        slope = (le[-1] - le[-3]) / (ns[-1] - ns[-3])
        return float(np.exp(le[-1] + slope * (n - ns[-1])))

    def circ_dur(self, b, tmpl, n):
        ns, du = self.dur_tab[(b, tmpl)]
        return float(np.interp(n, ns, du))

    def t_sv(self, tmpl, n):
        ns, tsv, tip, a, _ = self.sv_fit[tmpl]
        if n <= ns[-1]:
            return float(np.interp(n, ns, tsv))
        return float(2 ** np.polyval(a, n))

    def t_ip(self, tmpl, n):
        ns, tsv, tip, _, b = self.sv_fit[tmpl]
        if n <= ns[-1]:
            return float(np.interp(n, ns, tip))
        return float(2 ** np.polyval(b, n))

    # ---------------------------------------------------------------- queue statistics
    def wait_mean(self, mu):
        return mu * math.exp(self.sig_q ** 2 / 2)

    def wait_std(self, mu):
        s2 = self.sig_q ** 2
        return mu * math.sqrt((math.exp(s2) - 1) * math.exp(s2))

    def wait_cvar(self, mu, alpha):
        from scipy.stats import norm
        z = norm.ppf(alpha)
        return mu * math.exp(self.sig_q ** 2 / 2) * norm.cdf(self.sig_q - z) / (1 - alpha)

    def wait_quantile(self, mu, q):
        from scipy.stats import norm
        return mu * math.exp(self.sig_q * norm.ppf(q))


class Env:
    """One stochastic realisation of the hybrid platform (queues, outages, drift), shared by all methods (CRN)."""

    def __init__(self, plat, seed, horizon_h=720, drift_scale=1.0, jump_rate_per_h=1 / 24.0, queue_scale=None):
        self.p = plat
        rng = np.random.default_rng(1_000_003 + seed)
        self.dt = 60.0
        N = int(horizon_h * 3600 / self.dt) + 2
        self.N = N
        hrs = self.dt / 3600.0
        qs = plat.queue_scale if queue_scale is None else queue_scale
        # global load regime: Markov chain over {0.5, 1, 2.5}, mean dwell 2 h (vectorised)
        levels = np.array([0.5, 1.0, 2.5]); P_stay = math.exp(-self.dt / 7200.0)
        sw = rng.random(N) > P_stay
        step = np.where(sw, rng.integers(1, 3, N), 0)
        load = levels[(1 + np.cumsum(step)) % 3]
        tgrid = np.arange(N) * self.dt
        idx = np.arange(N)
        self.mu, self.cal, self.e_true, self.e_pub = {}, {}, {}, {}
        mu_d, sd_d = 0.005 * drift_scale, 0.03 * drift_scale
        for b in BACKENDS:
            ph = rng.uniform(0, 24)
            diurnal = 1 + 0.5 * np.sin(2 * np.pi * (tgrid / 3600.0 + ph) / 24.0)
            self.mu[b] = qs * plat.mu0[b] * load * diurnal
            c0 = rng.uniform(0, 24 * 3600)
            starts = np.arange(c0 - 24 * 3600, horizon_h * 3600 + 24 * 3600, 24 * 3600)
            self.cal[b] = [(s0, s0 + 2700.0) for s0 in starts]
            ends = np.array(sorted({int((s0 + 2700.0) // self.dt) for s0 in starts if 0 <= (s0 + 2700.0) // self.dt < N}))
            # segment id of each step (segment starts at a calibration end; segment 0 before the first one)
            seg = np.searchsorted(ends, idx, side='right')
            seg_start = np.concatenate([[0], ends])[seg]
            inc = mu_d * hrs + sd_d * math.sqrt(hrs) * rng.standard_normal(N)
            jmp = (rng.random(N) < jump_rate_per_h * drift_scale * hrs) * rng.uniform(0.2, 0.7, N)
            x = inc + jmp
            x[seg_start == idx] = 0.0
            cs = np.cumsum(x)
            base = cs[seg_start]
            reset = rng.normal(0, 0.05, len(ends) + 1)[seg]
            loge = reset + cs - base
            # publications: at each calibration end and every 6 h after it
            pub_idx = seg_start + 360 * ((idx - seg_start) // 360)
            self.e_true[b] = np.exp(loge)
            self.e_pub[b] = np.exp(loge[pub_idx])
        self.seed = seed
        self.dur_rng = np.random.default_rng(seed * 104729 + 17)
        self.e_min = {b: float(self.e_true[b].min()) for b in BACKENDS}

    def prepare(self, n, cv=0.1):
        """Common random numbers keyed by task id: every method sees the same duration noise, the same
        queue-wait deviate for a given (backend, task) and the same execution jitter."""
        rng = np.random.default_rng(self.seed * 7777 + 13)
        self.Zw = rng.standard_normal((len(BACKENDS), n))
        self.dnoise = np.clip(np.exp(rng.normal(-cv * cv / 2, cv, n)), 0.8, 1.3)
        self.qjit = rng.uniform(0.98, 1.05, (n, 2))
        self.wnoise = rng.standard_normal((n, 2))

    def _i(self, t):
        return min(self.N - 1, max(0, int(t // self.dt)))

    def mu_at(self, b, t):
        return float(self.mu[b][self._i(t)])

    def e_at(self, b, t):
        return float(self.e_true[b][self._i(t)])

    def epub_at(self, b, t):
        return float(self.e_pub[b][self._i(t)])

    def sample_wait(self, b, t, j=None):
        z = self.Zw[BACKENDS.index(b), j] if j is not None else self.dur_rng.normal()
        return self.mu_at(b, t) * math.exp(self.p.sig_q * z)

    def outage_end(self, b, t):
        for s0, s1 in self.cal[b]:
            if s0 <= t < s1:
                return s1
        return None


# ===================================================================== execution modes (CAMS)
class Mode:
    __slots__ = ('kind', 'b', 'mit', 'shots', 'occ', 'cost', 'F', 'idx', 'nc', 'reps', 'occ_by', 'stats', 'sw')

    def __init__(self, kind, b, mit, shots, occ, cost, F, nc, reps=None, sw=None):
        self.kind, self.b, self.mit, self.shots, self.occ, self.cost, self.F, self.nc = kind, b, mit, shots, occ, cost, F, nc
        self.sw = sw          # total witness shots of the job (None: default monitoring witness)
        self.idx = -1
        self.reps = reps if reps is not None else ([self] if kind == 'Q' else [])
        self.occ_by = {'N': occ} if kind == 'E' else ({b: occ} if kind == 'Q' else {})
        self.stats = None

    def __repr__(self):
        return f"Mode({self.kind},{self.b},{self.mit},s={self.shots},occ={self.occ:.0f},c={self.cost:.0f})"


def n_eff(task):
    return min(task['n'], 6) if task['cls'] == 'V' else task['n']   # light-cone for local observables


def task_fidelity(plat, b, task, e):
    """Fidelity of task's circuits on backend b under error multiplier e (F = ESP0^e)."""
    tmpl = TEMPLATE[task['cls']]
    return plat.esp0(b, tmpl, n_eff(task)) ** e


def drift_factor(plat, mu_b, elapsed_s):
    """Lemma 2: multiplier on the last observed error multiplier such that, with probability >= 1-eta_d, the
    multiplier stays below the bound during the whole window [t_obs, t_obs+h]; the window covers the elapsed
    time since the observation, the (1-eta_q) quantile of the queue wait and a margin for execution."""
    from scipy.stats import norm
    h = max(0.0, (elapsed_s + plat.wait_quantile(mu_b, 1.0 - plat.eta_q) + plat.drift_margin) / 3600.0)
    z = norm.ppf(1.0 - plat.eta_d / 2.0)          # reflection principle: uniform over the window
    return math.exp(plat.mu_d * h + z * plat.sd_d * math.sqrt(h))


_DGRID = np.geomspace(1e-4, 0.5, 80)


def shots_plan(plat, task, b, F, mit, witness=False):
    """Precision-guaranteed shot plan (Lemma 1). Returns (shots per circuit, witness shots or None), or None.

    * sampling tasks: multiplicative Chernoff bound + union bound over the task's circuits;
    * raw estimation: Hoeffding + union bound, bias (1-F)*kappa;
    * rescaled estimation with in-job witness: the witness fidelity error Delta (Hoeffding on the witness,
      one more event in the union bound) enters both the bias and the rescaled range; Delta, and hence the
      witness shot count, is chosen to minimise the job's total shot volume;
    * rescaled estimation without witness (baselines as practised): bias model only, no guarantee."""
    cls, eps, kap, nc = task['cls'], task['eps'], task['kappa'], task['nc']
    if cls == 'S':
        if F < plat.f_floor or mit:
            return None
        psi = F + (1 - F) * plat.r0
        nv = task['nv']
        L = math.log(nc / plat.eta)
        s = math.ceil((nv + L + math.sqrt(L * L + 2 * L * nv)) / psi)
        return (s, None) if s <= plat.max_shots else None
    R0 = 1.0 if cls == 'K' else 2.0
    if not mit:
        beta = (1 - F) * kap
        margin = eps - beta
        if margin <= 0.1 * eps:
            return None
        s = math.ceil(R0 * R0 * math.log(2 * nc / plat.eta) / (2 * margin * margin))
        return (s, None) if s <= plat.max_shots else None
    if not witness:
        beta = plat.xi[b] * (1 - F) * kap; R = R0 / F
        margin = eps - beta
        if margin <= 0.1 * eps:
            return None
        s = math.ceil(R * R * math.log(2 * nc / plat.eta) / (2 * margin * margin))
        return (s, None) if s <= plat.max_shots else None
    Lw = math.log(2 * (nc + 1) / plat.eta)
    pmix = 2.0 ** (-n_eff(task))
    best = None
    for D in _DGRID * F:
        Fl = F - D
        beta = kap * (plat.xi[b] * F * (1 - F) + D) / Fl          # eq. (4) with |F_hat - F| <= D
        margin = eps - beta
        if margin <= 0.1 * eps:
            continue
        R = R0 / Fl
        s = math.ceil(R * R * Lw / (2 * margin * margin))
        if s > plat.max_shots:
            continue
        sw = math.ceil(Lw / (2 * (D * (1 - pmix)) ** 2))
        vol = nc * s + sw
        if best is None or vol < best[0]:
            best = (vol, s, sw)
    return None if best is None else (best[1], best[2])


def shots_for(plat, task, b, F, mit, witness=False):
    r = shots_plan(plat, task, b, F, mit, witness)
    return None if r is None else r[0]


def witness_shots(plat, task, shots, sw=None):
    """Total shots of the two known-answer witness circuits: the precision-optimised count for rescaled
    modes (sw), otherwise a monitoring witness carrying ~witness_frac of the job's shot volume."""
    if sw is not None:
        return int(sw)
    return 2 * max(1000, int(plat.witness_frac * shots * task['nc'] / 2))


def qpu_exec_time(plat, task, b, shots, witness, sw=None):
    tmpl = TEMPLATE[task['cls']]
    per_shot = plat.circ_dur(b, tmpl, task['n']) + plat.rep
    t = plat.t_job + task['nc'] * (plat.t_circ + shots * per_shot)
    if witness:
        t += 2 * plat.t_circ + witness_shots(plat, task, shots, sw) * per_shot
    return t


def emulation_time(plat, task):
    n, cls = task['n'], task['cls']
    if n > 29:
        return None
    tmpl = TEMPLATE[cls]
    tsv, tip = plat.t_sv(tmpl, n), plat.t_ip(tmpl, n)
    if cls == 'K':
        br, bc = task['br'], task['bc']
        svb = (2 ** n) * 16.0
        c = int(0.8 * plat.mem // svb) - 1
        if c < 1:
            return None
        nsv = br + bc if c >= bc else bc + br * math.ceil(bc / c)
        return nsv * tsv + br * bc * tip
    if cls == 'V':
        return (2 * task['P'] + 1) * (tsv + tip)
    return task['nc'] * 1.2 * tsv


_Z = np.random.default_rng(20261103).standard_normal((512, len(BACKENDS)))
_BIDX = {b: k for k, b in enumerate(BACKENDS)}


def lat_stats(plat, md, mu, backlog=None, cancel_lat=None):
    """Monte-Carlo latency statistics of a mode under log-normal queue waits (common random draws).
    Returns (mean, var, exp_cost, exp_occ_by_type). backlog: dict b -> earliest time (relative) our device is free."""
    if md.kind == 'E':
        return md.occ, 0.0, 0.0, {'N': md.occ}
    if cancel_lat is None:
        cancel_lat = getattr(plat, 'cancel_lat', 5.0)
    up, dn = plat.wan_up, plat.wan_dn
    st = []
    for r in md.reps:
        w = mu[r.b] * np.exp(plat.sig_q * _Z[:, _BIDX[r.b]])
        s0 = up + w
        if backlog is not None:
            s0 = np.maximum(s0, backlog.get(r.b, 0.0))
        st.append(s0)
    st = np.array(st)
    if len(md.reps) == 1:
        fin = st[0] + md.reps[0].occ + dn
        return float(fin.mean()), float(fin.var()), md.reps[0].occ, {md.reps[0].b: md.reps[0].occ}
    win = np.argmin(st, axis=0)
    occs = np.array([r.occ for r in md.reps])
    tstart = st[win, np.arange(st.shape[1])]
    fin = tstart + occs[win] + dn
    srt = np.sort(st, axis=0)
    double = (srt[1] - srt[0]) < cancel_lat
    cost = occs[win] + double * np.where(win == 0, occs[1], occs[0])
    occ_by = {}
    for k, r in enumerate(md.reps):
        occ_by[r.b] = float(np.mean(((win == k) | double) * r.occ))
    return float(fin.mean()), float(fin.var()), float(cost.mean()), occ_by


def add_hedged(plat, modes, mu, max_pairs=6):
    """Composite queue-hedged modes: replicas on two backends, first-to-start wins, rest cancelled."""
    singles = sorted([m for m in modes if m.kind == 'Q'], key=lambda m: m.occ)
    out = []
    for a in range(len(singles)):
        for c in range(a + 1, len(singles)):
            r1, r2 = singles[a], singles[c]
            h = Mode('H', (r1.b, r2.b), None, 0, 0.0, 0.0, min(r1.F, r2.F), r1.nc, reps=[r1, r2])
            mean, var, cost, occ_by = lat_stats(plat, h, mu)
            h.cost = cost; h.occ_by = occ_by; h.occ = sum(occ_by.values()); h.stats = (mean, var)
            out.append(h)
    out.sort(key=lambda m: m.stats[0])
    return modes + out[:max_pairs]


def synthesize_modes(plat, task, e_plan, witness=False, emulation=True, mitigation=True, prune=True, hedge_mu=None):
    """CAMS: enumerate precision-feasible execution modes for a quantum task.

    e_plan: dict backend -> error multiplier used for planning (point estimate or upper confidence bound).
    Returns a list of Mode objects (Pareto-pruned on (occupancy, cost)).
    """
    modes = []
    for b in BACKENDS:
        F = task_fidelity(plat, b, task, e_plan[b])
        for mit in ((False, True) if (mitigation and task['cls'] != 'S') else (False,)):
            r = shots_plan(plat, task, b, F, mit, witness)
            if r is None:
                continue
            s, sw = r
            t = qpu_exec_time(plat, task, b, s, witness, sw)
            modes.append(Mode('Q', b, mit, s, t, t, F, task['nc'], sw=sw))
    if emulation:
        te = emulation_time(plat, task)
        if te is not None:
            modes.append(Mode('E', None, False, 0, te, 0.0, 1.0, task['nc']))
    if prune:
        # per backend keep the faster of raw/mitigated (equal cost = occupancy)
        best = {}
        for md in modes:
            key = md.b if md.kind == 'Q' else 'E'
            if key not in best or md.occ < best[key].occ:
                best[key] = md
        modes = list(best.values())
    if hedge_mu is not None:
        modes = add_hedged(plat, modes, hedge_mu)
    for i, md in enumerate(modes):
        md.idx = i
    return modes


def violation_prob(plat, task, mode, F_true, F_used):
    """Probability that the delivered estimate violates the task's precision target."""
    from scipy.stats import norm
    if mode.kind == 'E':
        return 0.0
    if task['cls'] == 'S':
        psi = F_true + (1 - F_true) * plat.r0
        mean, sd = mode.shots * psi, math.sqrt(mode.shots * psi * (1 - psi))
        p1 = norm.cdf((task['nv'] - mean) / max(sd, 1e-9))
        return float(1 - (1 - p1) ** task['nc'])
    kap, eps = task['kappa'], task['eps']
    R0 = 1.0 if task['cls'] == 'K' else 2.0
    if mode.mit:
        beta = abs(F_true / F_used - 1.0) * kap + plat.xi[mode.b] * (1 - F_true) * kap
        R = R0 / F_used
    else:
        beta = (1 - F_true) * kap
        R = R0
    sd = R / (2.0 * math.sqrt(mode.shots))
    p1 = norm.sf((eps - beta) / sd) + norm.sf((eps + beta) / sd)
    p1 = min(1.0, p1)
    return float(1 - (1 - p1) ** task['nc'])
