import sys, time, json
import numpy as np
from qdag.model import Platform, Env
from qdag.workflows import make
from qdag.sim import Simulator
from qdag.policies import make_policy, reference_budget, lower_bound

METHODS = ['HEFT-Q', 'PEFT-Q', 'SHEFT-Q', 'QG-Dyn', 'FG-Dyn', 'CF-Dyn', 'Q-DAGFlow']

def run(family, scale, seed, method, m_nodes=128, budget_factor=1.0, plat_kw=None, env_kw=None, qd_kw=None, wf=None, cams=False):
    plat = Platform(m_nodes=m_nodes, **(plat_kw or {}))
    env = Env(plat, seed, **(env_kw or {}))
    wf = wf or make(family, scale, seed)
    if method == 'LB':
        t0 = time.perf_counter(); T = lower_bound(wf, plat, env)
        return dict(family=family, scale=scale, seed=seed, method='LB', makespan=T, n_tasks=wf.n, n_q=int(wf.isq.sum()), wall=time.perf_counter() - t0)
    B = 1e15 if budget_factor is None else budget_factor * reference_budget(wf, plat, env)
    t0 = time.perf_counter()
    pol = make_policy(method, wf, plat, env, budget=B, cams=cams, **(qd_kw or {}))
    sim = Simulator(wf, plat, env, pol)
    res = sim.run()
    res.update(family=family, scale=scale, seed=seed, method=method, n_tasks=wf.n, n_q=int(wf.isq.sum()),
               budget=B, wall=time.perf_counter() - t0, sched_time=res['policy_time'] + getattr(pol, 'plan_time', 0.0),
               n_replans=getattr(pol, 'n_replans', 0), cams=cams)
    if getattr(pol, 'lp_info', None):
        res.update(T_lp=pol.lp_info[0]['T_lp'], U_plan=pol.lp_info[0]['U'], lp_cost=pol.lp_info[0]['cost'])
    return res

if __name__ == '__main__':
    fam, sc, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    for m in METHODS:
        r = run(fam, sc, seed, m)
        print('%-10s ms=%8.0f  cost=%8.0f/%8.0f viol=%.4f nQ=%4d nE=%4d util=%.2f sched=%.2fs wall=%.1fs complete=%s rp=%d' % (
            m, r['makespan'], r['qpu_cost'], r['budget'], r['viol_mean'], r['n_qpu'], r['n_emul'], r['util'], r['sched_time'], r['wall'], r['complete'], r['n_replans']))
