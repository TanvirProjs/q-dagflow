"""Scheduling policies: static list schedulers (HEFT-Q, PEFT-Q, SHEFT-Q), dynamic baselines
(QG-Dyn, FG-Dyn, CF-Dyn) and Q-DAGFlow (CAMS + LP-MA + RADAR + drift-triggered reconfiguration)."""
import heapq, math, time
import numpy as np
from .model import BACKENDS, synthesize_modes, task_fidelity, shots_plan, qpu_exec_time, Mode, lat_stats, drift_factor
from . import lp as LP


def forced_mode(plat, task, e):
    """Last-resort mode when no precision-feasible mode exists: best-fidelity QPU at max shots."""
    b = max(BACKENDS, key=lambda b: task_fidelity(plat, b, task, e[b]))
    F = task_fidelity(plat, b, task, e[b])
    s = plat.max_shots
    t = qpu_exec_time(plat, task, b, s, False)
    md = Mode('Q', b, task['cls'] != 'S', s, t, t, F, task['nc']); md.idx = 0
    return md


class Base:
    name = 'base'
    cams = False          # iso-precision mode: baselines use CAMS menus (LCB shots + in-job witness rescaling)

    def __init__(self, wf, plat, env):
        self.wf, self.p, self.env = wf, plat, env
        self.comm = lambda i, j: plat.lat + wf.ebytes[(i, j)] / plat.bw

    def info_e(self, t):
        """Error multipliers used by a baseline for planning shots at time t."""
        e = {b: self.env.epub_at(b, t) for b in BACKENDS}
        if self.cams:
            for b in BACKENDS:   # unknown staleness of the published calibration: assume 3 h
                e[b] *= drift_factor(self.p, self.env.mu_at(b, t), 3 * 3600.0)
        return e

    def fmode(self):
        return 'witness' if self.cams else 'pub'

    def on_start(self, sim):
        pass

    def on_qpu_done(self, sim, j, t):
        pass

    # ---------------- helpers
    def pub_e(self, t):
        return {b: self.env.epub_at(b, t) for b in BACKENDS}

    def menu(self, j, e, witness=None, emulation=True):
        witness = self.cams if witness is None else witness
        m = synthesize_modes(self.p, self.wf.q[j], e, witness=witness, emulation=emulation)
        return m if m else [forced_mode(self.p, self.wf.q[j], e)]

    def qlat(self, md, mu, how='mean', alpha=0.9):
        if md.kind == 'E':
            return md.occ
        p = self.p
        if how == 'mean':
            w = p.wait_mean(mu)
        elif how == 'meanstd':
            w = p.wait_mean(mu) + p.wait_std(mu)
        else:
            w = p.wait_cvar(mu, alpha)
        return p.wan_up + w + md.occ + p.wan_dn

    def upward_rank(self, w, R=None):
        wf = self.wf
        R = wf.topo if R is None else R
        inR = set(R)
        rank = {}
        for j in reversed(R):
            best = 0.0
            for s in wf.succ[j]:
                if s in inR:
                    v = self.comm(j, s) + rank[s]
                    if v > best:
                        best = v
            rank[j] = w[j] + best
        return rank

    def shots_now(self, j, md, e):
        """Recompute the shot plan (shots, witness shots) for a chosen (backend, mitigation) at submission."""
        task = self.wf.q[j]
        F = task_fidelity(self.p, md.b, task, e[md.b])
        return shots_plan(self.p, task, md.b, F, md.mit, self.cams)


# =============================================================== static list schedulers
class StaticList(Base):
    """HEFT-Q / SHEFT-Q / PEFT-Q: plan once at t=0 with published calibration and mean (or mean+std) waits."""

    def __init__(self, wf, plat, env, variant='heft'):
        super().__init__(wf, plat, env)
        self.variant = variant
        self.name = {'heft': 'HEFT-Q', 'sheft': 'SHEFT-Q', 'peft': 'PEFT-Q'}[variant]
        t0 = time.perf_counter()
        self._plan()
        self.plan_time = time.perf_counter() - t0

    def _plan(self):
        wf, p, env = self.wf, self.p, self.env
        n, m = wf.n, p.m
        e0 = self.info_e(0.0)
        mu0 = {b: env.mu_at(b, 0.0) for b in BACKENDS}
        how = 'meanstd' if self.variant == 'sheft' else 'mean'
        self.menus = {j: self.menu(j, e0) for j in range(n) if wf.isq[j]}
        lat = {j: [self.qlat(md, mu0[md.b] if md.kind == 'Q' else 0, how) for md in self.menus[j]] for j in self.menus}
        wbar = {j: (float(np.mean(lat[j])) if wf.isq[j] else wf.dur[j]) for j in range(n)}
        if self.variant == 'peft':
            oct_, prio = self._oct(lat)
        else:
            prio = self.upward_rank(wbar)
        avail = np.zeros(m); availq = {b: 0.0 for b in BACKENDS}
        AFT = np.zeros(n); proc = np.full(n, -2)
        self.assign = [None] * n
        self.order = [[] for _ in range(m)]
        indeg = np.array([len(x) for x in wf.pred])
        heap = [(-prio[j], j) for j in range(n) if indeg[j] == 0]
        heapq.heapify(heap)
        while heap:
            _, j = heapq.heappop(heap)
            preds = wf.pred[j]
            Mall = max([AFT[i] + self.comm(i, j) for i in preds], default=0.0)
            est = np.maximum(avail, Mall)
            for pn in {int(proc[i]) for i in preds if proc[i] >= 0}:
                r = max([AFT[i] + (0.0 if proc[i] == pn else self.comm(i, j)) for i in preds], default=0.0)
                est[pn] = max(avail[pn], r)
            opts = []   # (score, eft, kind, node/b, mode)
            def node_opt(d, md, cls_key):
                pnode = int(np.argmin(est))
                eft = est[pnode] + d
                sc = eft + (oct_[j][cls_key] if self.variant == 'peft' else 0.0)
                opts.append((sc, eft, 'N', pnode, md))
            if not wf.isq[j]:
                node_opt(wf.dur[j], None, 'N')
            else:
                rc = max([AFT[i] + self.comm(i, j) for i in preds], default=0.0)
                for k, md in enumerate(self.menus[j]):
                    if md.kind == 'E':
                        node_opt(md.occ, md, 'N')
                    else:
                        w = lat[j][k] - md.occ - p.wan_dn - p.wan_up
                        st = max(rc + p.wan_up + w, availq[md.b])
                        eft = st + md.occ + p.wan_dn
                        sc = eft + (oct_[j][md.b] if self.variant == 'peft' else 0.0)
                        opts.append((sc, eft, 'Q', md.b, md, st))
            o = min(opts, key=lambda z: z[0])
            if o[2] == 'N':
                pn = o[3]; d = o[1] - est[pn]
                avail[pn] = o[1]; AFT[j] = o[1]; proc[j] = pn
                self.assign[j] = ('N', pn, o[4]); self.order[pn].append(j)
            else:
                availq[o[3]] = o[5] + o[4].occ; AFT[j] = o[1]; proc[j] = -1
                self.assign[j] = ('Q', o[4])
            for s in wf.succ[j]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    heapq.heappush(heap, (-prio[s], s))
        self.ptr = [0] * m
        self.ready = np.zeros(n, bool)
        self.floating = []

    def _oct(self, lat):
        wf = self.wf
        classes = {}
        wcls = {}
        for j in range(wf.n):
            if not wf.isq[j]:
                classes[j] = ['N']; wcls[j] = {'N': wf.dur[j]}
            else:
                d = {}
                for k, md in enumerate(self.menus[j]):
                    key = 'N' if md.kind == 'E' else md.b
                    d[key] = min(d.get(key, math.inf), lat[j][k])
                classes[j] = list(d.keys()); wcls[j] = d
        OCT = {}
        for i in reversed(wf.topo):
            OCT[i] = {}
            for k in classes[i]:
                best = 0.0
                for s in wf.succ[i]:
                    v = min(OCT[s][k2] + wcls[s][k2] + (0.0 if k2 == k else self.comm(i, s)) for k2 in classes[s])
                    best = max(best, v)
                OCT[i][k] = best
        prio = {i: float(np.mean(list(OCT[i].values()))) for i in range(wf.n)}
        return OCT, prio

    def _head_try(self, sim, pn, t):
        if pn not in sim.free_nodes:
            return
        o = self.order[pn]
        k = self.ptr[pn]
        if k < len(o) and self.ready[o[k]]:
            j = o[k]; self.ptr[pn] += 1
            md = self.assign[j][2]
            sim.mode[j] = md
            sim.start_on_node(j, pn, t, md.occ if md is not None else self.wf.dur[j])

    def on_ready(self, sim, j, t):
        a = self.assign[j]
        if a[0] == 'Q':
            self._submit(sim, j, a[1], t)
        else:
            self.ready[j] = True
            self._head_try(sim, a[1], t)

    def _submit(self, sim, j, md, t):
        e = self.info_e(t)
        r = self.shots_now(j, md, e)
        s, sw = r if r is not None else (None, None)
        if s is None:
            alt = [x for x in self.menu(j, e) if x.kind == 'Q']
            if not alt:
                if any(x.kind == 'E' for x in self.menus[j]):
                    heapq.heappush(self.floating, (0, j)); return
                alt = [forced_mode(self.p, self.wf.q[j], e)]
            md = max(alt, key=lambda x: x.F); s = md.shots; sw = md.sw
        sim.submit_qpu(j, md, t, s, self.fmode(), self.cams, sw=sw)

    def dispatch(self, sim, t):
        for pn in list(sim.free_nodes):
            self._head_try(sim, pn, t)
        while self.floating and sim.free_nodes:
            _, j = heapq.heappop(self.floating)
            md = [x for x in self.menus[j] if x.kind == 'E'][0]
            pn = sim.best_node(j, t); sim.mode[j] = md
            sim.start_on_node(j, pn, t, md.occ)


# =============================================================== dynamic baselines
class Dynamic(Base):
    """Common machinery: priority = upward rank (mean latencies); classical tasks go to the free node with
    best data locality; quantum tasks pick a mode when they become ready."""

    def __init__(self, wf, plat, env, rule='qg', w=0.5):
        super().__init__(wf, plat, env)
        self.rule = rule
        self.w = w        # FT-Dyn: weight of normalised infidelity vs. normalised completion time
        self.name = {'qg': 'QG-Dyn', 'fg': 'FG-Dyn', 'cf': 'CF-Dyn', 'qgh': 'QGH-Dyn', 'ft': 'FT-Dyn'}[rule]
        t0 = time.perf_counter()
        e0 = self.info_e(0.0); mu0 = {b: env.mu_at(b, 0.0) for b in BACKENDS}
        w = {}
        for j in range(wf.n):
            if wf.isq[j]:
                mm = self.menu(j, e0)
                w[j] = float(np.mean([self.qlat(md, mu0[md.b] if md.kind == 'Q' else 0) for md in mm]))
            else:
                w[j] = wf.dur[j]
        self.rank = self.upward_rank(w)
        self.heap = []
        self.plan_time = time.perf_counter() - t0

    def on_ready(self, sim, j, t):
        if not self.wf.isq[j]:
            heapq.heappush(self.heap, (-self.rank[j], j, None)); return
        e = self.info_e(t)
        if self.rule == 'qgh':   # queue-greedy with the same hedged composite modes as Q-DAGFlow
            mu = {b: self.env.mu_at(b, t) for b in BACKENDS}
            mm = synthesize_modes(self.p, self.wf.q[j], e, witness=self.cams, emulation=True, hedge_mu=mu) or [forced_mode(self.p, self.wf.q[j], e)]
        else:
            mm = self.menu(j, e)
        md = self.choose(sim, j, t, mm)
        if md.kind == 'E':
            heapq.heappush(self.heap, (-self.rank[j], j, md))
        else:
            sim.submit_qpu(j, md, t, md.shots, self.fmode(), self.cams)

    def est_finish(self, sim, j, md, t, how='mean', alpha=0.9):
        p = self.p
        if md.kind == 'E':
            return t + sim.node_wait_estimate(t, len(self.heap)) + md.occ
        mu = self.env.mu_at(md.b, t)
        w = p.wait_mean(mu) if how == 'mean' else p.wait_cvar(mu, alpha)
        arr = t + sim.upload_time(j) + w
        backlog = sim.dev_known_free(md.b, t) + 0.5 * sum(sim.inflight[md.b].values())
        return max(arr, backlog) + md.occ + p.wan_dn

    def choose(self, sim, j, t, mm):
        qm = [x for x in mm if x.kind == 'Q']
        em = [x for x in mm if x.kind == 'E']
        if self.rule == 'qg':
            return min(mm, key=lambda x: self.est_finish(sim, j, x, t))
        if self.rule == 'qgh':
            backlog = {b: sim.dev_known_free(b, t) - t + 0.5 * sum(sim.inflight[b].values()) for b in BACKENDS}
            mu = {b: self.env.mu_at(b, t) for b in BACKENDS}
            def fin(x):
                if x.kind == 'E':
                    return self.est_finish(sim, j, x, t)
                return t + lat_stats(self.p, x, mu, backlog)[0]
            return min(mm, key=fin)
        if self.rule == 'ft':    # scalarised fidelity-latency trade-off (Qonductor-style front selection)
            fins = {id(x): self.est_finish(sim, j, x, t) - t for x in mm}
            tmin = max(1e-9, min(fins.values()))
            imax = max(1e-12, max(1.0 - x.F for x in mm))
            return min(mm, key=lambda x: (1 - self.w) * fins[id(x)] / tmin + self.w * (1.0 - x.F) / imax)
        if self.rule == 'cf' and em:
            return em[0]
        if qm:
            return max(qm, key=lambda x: x.F)
        return em[0]

    def dispatch(self, sim, t):
        while sim.free_nodes and self.heap:
            _, j, md = heapq.heappop(self.heap)
            pn = sim.best_node(j, t)
            sim.mode[j] = md
            sim.start_on_node(j, pn, t, md.occ if md is not None else self.wf.dur[j])


# =============================================================== Q-DAGFlow
class QDAGFlow(Base):
    """CAMS (precision-guaranteed, drift-robust mode synthesis incl. queue-hedged composites)
    + LP-MA (LP relaxation, Markov/argmax rounding, budget repair, critical upgrade)
    + RADAR (path-quantile risk-aware dispatch with budget slack)
    + in-situ witness calibration and CUSUM-triggered reconfiguration."""
    name = 'Q-DAGFlow'

    def __init__(self, wf, plat, env, budget, z=1.2816, rho=(1 / 3, 1 / 3), lcb=True, witness=True,
                 use_lp=True, risk=True, switch=True, reconfig=True, emulation=True, hedge=True,
                 cusum_k=0.05, cusum_h=0.15, ewma=0.5, replan_gap=1200.0, max_replans=25):
        super().__init__(wf, plat, env)
        self.B = budget
        self.z, self.rho = z, rho
        self.lcb, self.witness, self.use_lp, self.risk = lcb, witness, use_lp, risk
        self.switch, self.reconfig, self.emulation, self.hedge = switch, reconfig, emulation, hedge
        self.ewma = ewma
        self.ck, self.ch = cusum_k, cusum_h
        self.replan_gap, self.max_replans = replan_gap, max_replans
        self.e_hat = {b: env.epub_at(b, 0.0) for b in BACKENDS}
        self.t_hat = {b: 0.0 for b in BACKENDS}
        self.last_pub = dict(self.e_hat)
        self.cus = {b: 0.0 for b in BACKENDS}
        self.heap = []
        self.committed = 0.0
        self.rem_plan = 0.0
        self.plan_cost, self.plan_key, self.vrest, self.rank = {}, {}, {}, {}
        self.dispatched = np.zeros(wf.n, bool)
        self.est_fin = np.zeros(wf.n)
        self.n_replans = 0; self.last_replan = -1e18
        self.lp_info = []
        self.plan_time = 0.0

    # ---------------- information state
    def refresh_pub(self, t):
        for b in BACKENDS:
            ep = self.env.epub_at(b, t)
            if ep != self.last_pub[b]:
                self.last_pub[b] = ep
                self.e_hat[b] = ep; self.t_hat[b] = t; self.cus[b] = 0.0

    def e_plan(self, t):
        out = {}
        for b in BACKENDS:
            e = self.e_hat[b]
            if self.lcb:
                e *= drift_factor(self.p, self.env.mu_at(b, t), t - self.t_hat[b])
            out[b] = e
        return out

    def menus_at(self, j, t, e=None):
        e = e or self.e_plan(t)
        mu = {b: self.env.mu_at(b, t) for b in BACKENDS} if self.hedge else None
        return self.menu_h(j, e, mu)

    def menu_h(self, j, e, mu):
        m = synthesize_modes(self.p, self.wf.q[j], e, witness=self.witness, emulation=self.emulation, hedge_mu=mu)
        return m if m else [forced_mode(self.p, self.wf.q[j], e)]

    def stats(self, md, t, backlog=None):
        mu = {b: self.env.mu_at(b, t) for b in BACKENDS}
        return lat_stats(self.p, md, mu, backlog)

    @staticmethod
    def key(md):
        if md.kind == 'E':
            return ('E',)
        return ('H',) + tuple(md.b) if md.kind == 'H' else ('Q', md.b)

    # ---------------- planning (LP-MA)
    def plan(self, sim, t):
        t0 = time.perf_counter()
        wf = self.wf
        R = [j for j in wf.topo if not self.dispatched[j]]
        if not R:
            return
        inR = set(R)
        e = self.e_plan(t)
        mu = {b: self.env.mu_at(b, t) for b in BACKENDS}
        menus = {j: self.menu_h(j, e, mu if self.hedge else None) for j in R if wf.isq[j]}
        st = {j: [lat_stats(self.p, md, mu) for md in menus[j]] for j in menus}
        lat = {j: [x[0] for x in st[j]] for j in menus}
        for j in menus:
            for md, x in zip(menus[j], st[j]):
                md.cost = x[2] if md.kind != 'E' else 0.0
                md.occ_by = x[3]
        dur = wf.dur
        release = {}
        for j in R:
            r = t
            for i in wf.pred[j]:
                if i not in inR:
                    f = sim.finish[i] if np.isfinite(sim.finish[i]) else max(t, self.est_fin[i])
                    r = max(r, f + self.comm(i, j))
            release[j] = r
        B_rem = max(0.0, self.B - self.committed)
        if self.use_lp:
            x, T, status = LP.solve_mode_lp(wf, R, menus, lat, dur, self.comm, self.p.m, B_rem, release)
            if x is None:
                cands = [{j: int(np.argmin([m.cost for m in menus[j]])) for j in menus}]
                T = None
            else:
                cands = [LP.argmax_round(x), LP.markov_round(x, lat, menus, self.p.m, *self.rho)]
            best = None
            pure = cands[-1] if x is not None else None
            if pure is not None and sum(menus[j][k].cost for j, k in pure.items()) <= B_rem + 1e-6:
                cands.append(('pure', pure))       # Theorem 1 plan, kept unmodified when it meets the budget
            for a in cands:
                if isinstance(a, tuple):
                    a = a[1]; c = sum(menus[j][k].cost for j, k in a.items())
                else:
                    a, c = LP.repair_budget(wf, R, a, menus, lat, dur, self.comm, B_rem, release)
                    a, c = LP.upgrade_critical(wf, R, a, menus, lat, dur, self.comm, B_rem, release)
                d = {j: (lat[j][a[j]] if wf.isq[j] else dur[j]) for j in R}
                occ = {j: (menus[j][a[j]].occ_by if wf.isq[j] else {'N': dur[j]}) for j in R}
                U = LP.graham_bound(wf, R, d, occ, self.comm, self.p.m, release)
                if best is None or U < best[0]:
                    best = (U, a, c)
            U, a, c = best
            self.lp_info.append(dict(t=t, T_lp=T, U=U, cost=c, B=B_rem, nR=len(R)))
        else:
            a = {j: int(np.argmin(lat[j])) for j in menus}
        for j in menus:
            md = menus[j][a[j]]
            self.plan_key[j] = self.key(md)
            self.plan_cost[j] = md.cost if self.use_lp else min(m.cost for m in menus[j])
        d = {j: (lat[j][a[j]] if wf.isq[j] else dur[j]) for j in R}
        v = {j: (st[j][a[j]][1] if wf.isq[j] else (0.1 * dur[j]) ** 2) for j in R}
        self.vrest.update(LP.path_variance(wf, R, d, v, self.comm, release))
        self.rem_plan = sum(self.plan_cost.get(j, 0.0) for j in R if wf.isq[j])
        self.rank.update(self.upward_rank(d, R))
        self.plan_time += time.perf_counter() - t0

    def on_start(self, sim):
        self.plan(sim, 0.0)

    # ---------------- RADAR dispatch
    def slack(self, j):
        return self.B - self.committed - (self.rem_plan - self.plan_cost.get(j, 0.0))

    def risk_score(self, sim, j, md, t):
        if md.kind == 'E':
            mean = sim.node_wait_estimate(t, len(self.heap)) + md.occ
            var = (0.1 * md.occ) ** 2
        else:
            backlog = {b: sim.dev_known_free(b, t) - t + 0.5 * sum(sim.inflight[b].values()) for b in BACKENDS}
            mean, var, _, _ = self.stats(md, t, backlog)
        if not self.risk:
            return mean
        V = self.vrest.get(j, 0.0)
        return mean + self.z * (math.sqrt(V + var) - math.sqrt(V))

    def on_ready(self, sim, j, t):
        self.refresh_pub(t)
        self.dispatched[j] = True
        if not self.wf.isq[j]:
            self.est_fin[j] = t + self.wf.dur[j]
            heapq.heappush(self.heap, (-self.rank.get(j, 0.0), j, None)); return
        mm = self.menus_at(j, t)
        planned = [x for x in mm if self.key(x) == self.plan_key.get(j)]
        if self.switch or not planned:
            cap = self.plan_cost.get(j, 0.0) + max(0.0, self.slack(j))
            pk = self.plan_key.get(j, ('Q',))
            idle = len(sim.free_nodes) > len(self.heap)
            if planned and pk[0] == 'E':
                pool = [x for x in mm if x.kind == 'E']          # LP placed it on the CPU pool: keep the type
            else:
                pool = [x for x in mm if x.kind != 'E' or idle]  # QPU-planned: any backend/hedge; CPU only if a node idles
            cands = [x for x in pool if x.cost <= cap + 1e-6] or [min(mm, key=lambda x: x.cost)]
            md = min(cands, key=lambda x: (self.risk_score(sim, j, x, t), x.cost))
        else:
            md = planned[0]
        self.committed += md.cost
        self.rem_plan -= self.plan_cost.pop(j, 0.0)
        if md.kind == 'E':
            self.est_fin[j] = t + md.occ
            heapq.heappush(self.heap, (-self.rank.get(j, 0.0), j, md))
        else:
            self.est_fin[j] = t + self.stats(md, t)[0]
            sim.submit_qpu(j, md, t, None, 'witness' if self.witness else 'pub', self.witness)

    def dispatch(self, sim, t):
        while sim.free_nodes and self.heap:
            _, j, md = heapq.heappop(self.heap)
            pn = sim.best_node(j, t)
            sim.mode[j] = md
            sim.start_on_node(j, pn, t, md.occ if md is not None else self.wf.dur[j])

    # ---------------- in-situ calibration + change-point-triggered reconfiguration
    def on_qpu_done(self, sim, j, t):
        if not self.witness:
            return
        jb = sim.job[j]; rp = jb['reps'][jb['winner']]; b = rp['b']
        F0 = task_fidelity(self.p, b, self.wf.q[j], 1.0)
        if F0 >= 1.0:
            return
        e_obs = math.log(max(1e-9, rp['F_obs'])) / math.log(F0)
        rel = e_obs / self.e_hat[b] - 1.0
        self.cus[b] = max(0.0, self.cus[b] + rel - self.ck)
        self.e_hat[b] = (1 - self.ewma) * self.e_hat[b] + self.ewma * e_obs
        self.t_hat[b] = t
        if self.cus[b] > self.ch:
            self.cus[b] = 0.0
            if self.reconfig and self.n_replans < self.max_replans and t - self.last_replan > self.replan_gap:
                self.n_replans += 1; self.last_replan = t
                self.plan(sim, t)


def make_policy(name, wf, plat, env, budget=None, cams=False, **kw):
    if name in ('QGH-Dyn', 'FT-Dyn'):
        obj = Dynamic.__new__(Dynamic); obj.cams = cams
        obj.__init__(wf, plat, env, {'QGH-Dyn': 'qgh', 'FT-Dyn': 'ft'}[name], **kw)
        return obj
    if name in ('HEFT-Q', 'SHEFT-Q', 'PEFT-Q', 'QG-Dyn', 'FG-Dyn', 'CF-Dyn'):
        cls = StaticList if name in ('HEFT-Q', 'SHEFT-Q', 'PEFT-Q') else Dynamic
        arg = {'HEFT-Q': 'heft', 'SHEFT-Q': 'sheft', 'PEFT-Q': 'peft', 'QG-Dyn': 'qg', 'FG-Dyn': 'fg', 'CF-Dyn': 'cf'}[name]
        obj = cls.__new__(cls); obj.cams = cams
        obj.__init__(wf, plat, env, arg)
        return obj
    return QDAGFlow(wf, plat, env, budget, **kw)


def reference_budget(wf, plat, env):
    """B_ref: QPU-seconds of the default practice (best-fidelity backend for every QPU-feasible task)."""
    e0 = {b: env.epub_at(b, 0.0) for b in BACKENDS}
    tot = 0.0
    for j in range(wf.n):
        if wf.isq[j]:
            mm = [x for x in synthesize_modes(plat, wf.q[j], e0) if x.kind == 'Q']
            if mm:
                tot += max(mm, key=lambda x: x.F).cost
    return tot


def lower_bound(wf, plat, env):
    """Strict makespan lower bound valid for every policy and realisation: LP relaxation with zero queue
    waits, best fidelity over the horizon (fewest shots), minimum duration noise, no communication, no budget."""
    e = {b: env.e_min[b] for b in BACKENDS}
    menus = {}
    for j in range(wf.n):
        if wf.isq[j]:
            mm = synthesize_modes(plat, wf.q[j], e, witness=False)
            if not mm:
                mm = [forced_mode(plat, wf.q[j], e)]
            menus[j] = mm
    lat = {j: [(0.8 * md.occ if md.kind == 'E' else plat.wan_up + 0.98 * md.occ + plat.wan_dn) for md in menus[j]] for j in menus}
    for j in menus:
        for md in menus[j]:
            md.occ_by = {('N' if md.kind == 'E' else md.b): (0.8 if md.kind == 'E' else 0.98) * md.occ}
    dur = 0.8 * wf.dur
    x, T, st = LP.solve_mode_lp(wf, wf.topo, menus, lat, dur, lambda i, j: 0.0, plat.m, None, None, use_comm=False)
    return T
