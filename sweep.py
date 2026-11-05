import sys, json, time, itertools, os, traceback
from multiprocessing import Pool
from run_one import run

def job(args):
    fam, sc, seed, label, method, kw = args
    try:
        r = run(fam, sc, seed, method, **kw)
        r['label'] = label
        return r
    except Exception as e:
        return dict(family=fam, scale=sc, seed=seed, label=label, method=method, error=traceback.format_exc()[-800:])

def sweep(tasks, out, procs=2):
    t0 = time.time()
    res = []
    with Pool(procs) as pool:
        for k, r in enumerate(pool.imap_unordered(job, tasks, chunksize=1)):
            res.append(r)
            if 'error' in r:
                print('ERR', r['family'], r['scale'], r['seed'], r['label'], r['error'], flush=True)
            if (k + 1) % 50 == 0:
                print('%d/%d %.0fs' % (k + 1, len(tasks), time.time() - t0), flush=True)
                json.dump(res, open(out, 'w'))
    json.dump(res, open(out, 'w'))
    return res
