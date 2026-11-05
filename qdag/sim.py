"""Discrete-event simulator of a hybrid HPC-QPU platform executing a workflow under a policy.

QPU jobs may be submitted as several replicas (queue hedging); the first replica to start
execution wins and the others are cancelled (replicas starting within the cancellation
latency of the winner also execute and are billed)."""
import heapq, math, time
import numpy as np
from .model import BACKENDS, task_fidelity, violation_prob, qpu_exec_time, witness_shots

NODE_DONE, QPU_ARRIVE, QPU_DONE, RESULT, DEV_UP = range(5)


class Simulator:
    def __init__(self, wf, plat, env, policy, cancel_lat=None):
        self.wf, self.p, self.env, self.pol = wf, plat, env, policy
        n, m = wf.n, plat.m
        self.cancel_lat = getattr(plat, 'cancel_lat', 5.0) if cancel_lat is None else cancel_lat
        self.finish = np.full(n, np.inf)
        self.start_t = np.full(n, np.nan)
        self.loc = np.full(n, -2, dtype=np.int64)      # node id; -1 = cloud (QPU result)
        self.npred = np.array([len(x) for x in wf.pred])
        self.free_nodes = set(range(m))
        self.node_end = np.zeros(m)
        self.busy_time = 0.0
        self.ev, self.seq = [], 0
        self.dev_busy = {b: False for b in BACKENDS}
        self.dev_cur_end = {b: 0.0 for b in BACKENDS}
        self.dev_q = {b: [] for b in BACKENDS}
        self.dev_up_pending = {b: False for b in BACKENDS}
        self.inflight = {b: {} for b in BACKENDS}      # (j,r) submitted, not yet arrived -> occ
        self.mode = [None] * n
        self.job = {}
        self.qpu_cost = 0.0
        self.viol = []
        self.n_qpu = 0; self.n_emul = 0; self.n_hedged = 0; self.n_double = 0; self.n_cancel = 0
        self.qlat = []
        self.policy_time = 0.0
        self.done = 0
        env.prepare(n)

    # ---------------------------------------------------------------- utilities
    def push(self, t, typ, a, b=None):
        self.seq += 1
        heapq.heappush(self.ev, (t, self.seq, typ, a, b))

    def comm(self, i, j):
        return self.p.lat + self.wf.ebytes[(i, j)] / self.p.bw

    def data_ready(self, j, node):
        r = 0.0
        for i in self.wf.pred[j]:
            f = self.finish[i]
            if self.loc[i] != node:
                f += self.comm(i, j)
            if f > r:
                r = f
        return r

    def best_node(self, j, t):
        cands = {self.loc[i] for i in self.wf.pred[j] if self.loc[i] >= 0 and self.loc[i] in self.free_nodes}
        other = next(iter(self.free_nodes))
        cands.add(other)
        best, bt = None, math.inf
        for p in cands:
            r = self.data_ready(j, p)
            if r < bt:
                best, bt = p, r
        return best

    def noise(self, lo=0.8, hi=1.3, cv=0.1):
        return float(np.clip(math.exp(self.env.dur_rng.normal(-cv * cv / 2, cv)), lo, hi))

    def node_wait_estimate(self, t, k):
        nf = len(self.free_nodes)
        if k < nf:
            return 0.0
        busy_ends = np.sort(self.node_end[self.node_end > t])
        if not len(busy_ends):
            return 0.0
        return float(busy_ends[min(len(busy_ends) - 1, k - nf)] - t)

    # ---------------------------------------------------------------- node actions
    def start_on_node(self, j, node, t, dur):
        self.free_nodes.discard(node)
        st = max(t, self.data_ready(j, node))
        d = dur * float(self.env.dnoise[j])
        end = st + d
        self.start_t[j] = st
        self.node_end[node] = end
        self.busy_time += d
        self.loc[j] = node
        if self.wf.isq[j]:
            self.n_emul += 1
        self.push(end, NODE_DONE, j, node)

    # ---------------------------------------------------------------- QPU actions
    def upload_time(self, j):
        u = 0.0
        for i in self.wf.pred[j]:
            if self.loc[i] >= 0:
                u = max(u, self.comm(i, j))
        return u + self.p.wan_up

    def submit_qpu(self, j, md, t, shots=None, F_mode='pub', witness=False, rep_shots=None, sw=None):
        """Submit job j in mode md (single 'Q' or hedged 'H'); rep_shots overrides per-replica shots."""
        self.mode[j] = md
        reps = []
        for r, rm in enumerate(md.reps):
            s = rep_shots[r] if rep_shots is not None else (shots if (shots is not None and md.kind == 'Q') else rm.shots)
            w = sw if (sw is not None and md.kind == 'Q') else rm.sw
            reps.append(dict(b=rm.b, mit=rm.mit, shots=s, sw=w, mode=rm, state='sub'))
        self.job[j] = dict(reps=reps, F_mode=F_mode, witness=witness, t_sub=t, t_start=None, done=False)
        if len(reps) > 1:
            self.n_hedged += 1
        up = self.upload_time(j)
        for r, rp in enumerate(reps):
            b = rp['b']
            arr = t + up + self.env.sample_wait(b, t, j)
            self.inflight[b][(j, r)] = rp['mode'].occ
            self.push(arr, QPU_ARRIVE, (j, r), b)

    def dev_known_free(self, b, t):
        f = self.dev_cur_end[b] if self.dev_busy[b] else t
        for (_, _, (j, r)) in self.dev_q[b]:
            jb = self.job[j]
            if jb['t_start'] is None:
                f += jb['reps'][r]['mode'].occ
        return max(f, t)

    def _try_start(self, b, t):
        while not self.dev_busy[b] and self.dev_q[b]:
            oe = self.env.outage_end(b, t)
            if oe is not None:
                if not self.dev_up_pending[b]:
                    self.dev_up_pending[b] = True
                    self.push(oe, DEV_UP, b)
                return
            _, _, (j, r) = heapq.heappop(self.dev_q[b])
            jb = self.job[j]
            if jb['t_start'] is not None and t - jb['t_start'] >= self.cancel_lat:
                jb['reps'][r]['state'] = 'cancelled'; self.n_cancel += 1
                continue
            if jb['t_start'] is not None:
                self.n_double += 1
            else:
                jb['t_start'] = t
                self.start_t[j] = t
            rp = jb['reps'][r]
            task = self.wf.q[j]
            F_true = task_fidelity(self.p, b, task, self.env.e_at(b, t))
            if jb['F_mode'] == 'witness':
                sw = witness_shots(self.p, task, rp['shots'], rp['sw'])
                F_used = F_true * math.exp(float(self.env.wnoise[j, r]) * math.sqrt((1 - F_true) / (F_true * sw)))
            else:
                F_used = task_fidelity(self.p, b, task, self.env.epub_at(b, t))
            md = rp['mode']; s0 = md.shots; md.shots = rp['shots']
            rp['viol'] = violation_prob(self.p, task, md, F_true, F_used)
            md.shots = s0
            rp['F_obs'] = F_used; rp['F_true'] = F_true
            ex = qpu_exec_time(self.p, task, b, rp['shots'], jb['witness'], rp['sw']) * float(self.env.qjit[j, r])
            self.qpu_cost += ex
            self.n_qpu += 1
            rp['state'] = 'run'
            self.dev_busy[b] = True
            self.dev_cur_end[b] = t + ex
            self.push(t + ex, QPU_DONE, (j, r), b)
            return

    # ---------------------------------------------------------------- main loop
    def release(self, j, t):
        self.done += 1
        for s in self.wf.succ[j]:
            self.npred[s] -= 1
            if self.npred[s] == 0:
                t0 = time.perf_counter(); self.pol.on_ready(self, s, t); self.policy_time += time.perf_counter() - t0

    def run(self, t_limit=60 * 24 * 3600):
        t0 = time.perf_counter()
        self.pol.on_start(self)
        for j in range(self.wf.n):
            if self.npred[j] == 0:
                self.pol.on_ready(self, j, 0.0)
        self.pol.dispatch(self, 0.0)
        self.policy_time += time.perf_counter() - t0
        while self.ev:
            t = self.ev[0][0]
            if t > t_limit:
                break
            while self.ev and self.ev[0][0] == t:
                _, _, typ, a, b = heapq.heappop(self.ev)
                if typ == NODE_DONE:
                    self.finish[a] = t
                    self.free_nodes.add(b)
                    self.release(a, t)
                elif typ == QPU_ARRIVE:
                    self.inflight[b].pop(a, None)
                    j, r = a
                    jb = self.job[j]
                    if jb['t_start'] is not None and t - jb['t_start'] >= self.cancel_lat:
                        jb['reps'][r]['state'] = 'cancelled'; self.n_cancel += 1
                    else:
                        heapq.heappush(self.dev_q[b], (t, self.seq, a)); self.seq += 1
                        self._try_start(b, t)
                elif typ == QPU_DONE:
                    self.dev_busy[b] = False
                    j, r = a
                    jb = self.job[j]
                    if not jb['done']:
                        jb['done'] = True; jb['winner'] = r
                        self.viol.append(jb['reps'][r]['viol'])
                        self.qlat.append(t + self.p.wan_dn - jb['t_sub'])
                        self.push(t + self.p.wan_dn, RESULT, j)
                        t1 = time.perf_counter(); self.pol.on_qpu_done(self, j, t); self.policy_time += time.perf_counter() - t1
                    self._try_start(b, t)
                elif typ == RESULT:
                    self.finish[a] = t
                    self.loc[a] = -1
                    self.release(a, t)
                elif typ == DEV_UP:
                    self.dev_up_pending[a] = False
                    self._try_start(a, t)
            t1 = time.perf_counter(); self.pol.dispatch(self, t); self.policy_time += time.perf_counter() - t1
        ms = float(np.max(self.finish))
        v = np.array(self.viol) if self.viol else np.zeros(0)
        return dict(makespan=ms, complete=bool(self.done == self.wf.n), qpu_cost=self.qpu_cost,
                    viol_mean=float(v.mean()) if len(v) else 0.0, viol_expected=float(v.sum()),
                    n_qpu=self.n_qpu, n_emul=self.n_emul, n_hedged=self.n_hedged, n_double=self.n_double,
                    util=self.busy_time / (self.p.m * ms), policy_time=self.policy_time,
                    qlat_mean=float(np.mean(self.qlat)) if self.qlat else 0.0,
                    qlat_p95=float(np.percentile(self.qlat, 95)) if self.qlat else 0.0)
