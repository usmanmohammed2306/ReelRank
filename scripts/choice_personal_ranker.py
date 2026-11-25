"""Fit a viewer's time-ordered exposure-conditioned taste profile on SM-RS 2.0."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit, logsumexp
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from scripts import choice_content_ranker as base
from scripts import choice_fusion as fusion
from scripts.fetch_choice_osf import MANIFEST


WEIGHTS = (25, 50, 75, 100)
SEED = 5752026


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def unit_catalog(matrix):
    """Center reference coordinates, retaining zero vectors for missing rows."""
    known = np.any(matrix != 0, axis=1)
    result = np.zeros(matrix.shape, dtype=np.float64)
    result[known] = normalize(matrix[known] - matrix[known].mean(axis=0))
    return result


def bases_for(root, catalog, domain):
    import binpickle
    path = root / "artifacts" / f"implicit_mf_{domain}.bpk"
    if base.digest(path) != MANIFEST[path.name]["sha256"]:
        raise ValueError("Publisher item factors do not match their digest")
    predictor = binpickle.load(path).predictor
    n = len(catalog.popularity)
    vec = np.zeros((n, predictor.item_features_.shape[1]), dtype=np.float64)
    idx = np.asarray(predictor.item_index_, dtype=int)
    vec[idx] = predictor.item_features_
    # Reference-catalog decomposition does not observe SM-RS participant responses.
    reduced = TruncatedSVD(48, n_iter=6, random_state=SEED).fit_transform(vec)
    collaborative = unit_catalog(reduced)
    tags = unit_catalog(catalog.tags)
    raw = catalog.qualities.astype(float)
    quality = np.clip((raw - raw.mean(axis=0)) / np.maximum(raw.std(axis=0), 1e-4), -4, 4)
    quality /= np.sqrt(raw.shape[1])
    return (collaborative, tags, quality)


def history_features(vectors, candidates, exposed, chosen):
    """Build posterior scores from the visible catalog item representations.

    Each candidate and historical item occupies a single row. A previous
    display is exposure evidence even if the viewer has not responded yet.
    """
    q = vectors[candidates]
    a = vectors[exposed]
    p = vectors[chosen]
    n, k = len(a), len(p)
    if not n or not k:
        return np.zeros((len(q), 11), dtype=np.float64)
    a_mean = a.mean(axis=0)
    p_mean = p.mean(axis=0)
    z = a - a_mean
    # Shrink within-viewer covariance toward a sphere. Both settings are
    # supplied as features; the ranking model estimates their usefulness.
    gram = (z.T @ z) / n
    mean_variance = max(np.trace(gram) / len(gram), 1e-4)
    contrast = p_mean - a_mean
    diag = np.eye(len(gram)) * mean_variance
    qcenter = q - a_mean
    ridge = [qcenter @ np.linalg.solve(gram + strength * diag, contrast) * mean_variance
             for strength in (.2, 2.)]
    sim_p = q @ p.T
    sim_a = q @ a.T
    # A smooth local selection propensity, adjusted for what was offered.
    kernels = []
    for scale in (3., 8.):
        log_p = logsumexp(scale * sim_p, axis=1) - np.log(k)
        log_a = logsumexp(scale * sim_a, axis=1) - np.log(n)
        kernels.append((log_p - log_a) / scale)
    positive = sim_p.mean(axis=1)
    take = min(3, k)
    near = np.partition(sim_p, k - take, axis=1)[:, -take:].mean(axis=1)
    return np.column_stack([q @ contrast, positive, near, sim_p.max(axis=1), *ridge, *kernels,
                            q @ (p[-min(5, k):].mean(axis=0) - a_mean),
                            q @ contrast * np.log1p(k), ridge[0] * min(1., k / 20)])


def extra_rows(log, catalog, bases):
    """Access previous displays and clicks as distinct timestamped event streams."""
    views, clicks = [], []
    for row in log.itertuples():
        start = row.timestamp.value / 1e9
        is_rec = pd.notna(row.iteration)
        local = {int(o["itemId"]) - catalog.offset for o in row.impressions}
        views.append((start, is_rec, local))
        for o in row.impressions:
            if o["selected"] and np.isfinite(o["timestamp"]):
                clicks.append((max(float(o["timestamp"]), start), is_rec, int(o["itemId"]) - catalog.offset))
    views.sort(key=lambda v: v[0])
    clicks.sort()
    exposed, chosen = [set(), set()], [{}, {}]
    vi = ci = 0
    result = {}
    recs = log[log.iteration.notna()].sort_values(["timestamp", "impressionId"])
    for row in recs.itertuples():
        now = row.timestamp.value / 1e9
        while vi < len(views) and views[vi][0] < now:
            _, rec, items = views[vi]
            exposed[0].update(items)
            if rec:
                exposed[1].update(items)
            vi += 1
        while ci < len(clicks) and clicks[ci][0] < now:
            _, rec, item = clicks[ci]
            for scope in range(1 + int(rec)):
                chosen[scope].pop(item, None)
                chosen[scope][item] = None
            ci += 1
        q = np.array([int(o["itemId"]) - catalog.offset for o in row.impressions])
        coords = np.column_stack([bases[0][q, :16], bases[1][q, :16]])
        fields = []
        for scope in range(2):
            shown = np.array(sorted(exposed[scope]), dtype=int)
            picked = np.fromiter(chosen[scope], dtype=int)
            for space in bases:
                fields.append(history_features(space, q, shown, picked))
        result[int(row.impressionId)] = (coords, np.column_stack(fields))
    return result


def assemble(root, archive, users):
    cats = base.catalog(root, archive)
    cases = base.examples(root, cats, users)
    bases = {d:bases_for(root, cat, d) for d, cat in cats.items()}
    logs = pd.read_parquet(root / "data/behaviors.parquet", filters=[("userId", "in", users)])
    domains = {c.user:c.domain for c in cases}
    extra = {}
    for user, log in logs.groupby("userId"):
        domain = domains[int(user)]
        extra.update(extra_rows(log, cats[domain], bases[domain]))
    for c in cases:
        coords, past = extra[c.slate]
        # Catalog coordinates are independently oriented in movies and books.
        c.coords = np.column_stack([coords * (c.domain=="movies"), coords * (c.domain=="books")])
        c.personal = past
    return cases


def conditional_fit(x, y, test):
    """List-balanced cross entropy over the observed selected set."""
    active = (y.sum(axis=1) > 0) & (y.sum(axis=1) < y.shape[1])
    x = x[active]; y = y[active]
    x = x - x.mean(axis=1, keepdims=True)
    scale = x.std(axis=(0,1)); scale[scale<1e-5] = 1.
    x = np.clip(x/scale, -6, 6)
    target = y/y.sum(axis=1, keepdims=True)
    matrix = np.ascontiguousarray(x.reshape(-1,x.shape[2]))
    def fg(w):
        scores = (matrix @ w).reshape(target.shape)
        normalizer = logsumexp(scores,axis=1)
        prob = np.exp(scores-normalizer[:,None])
        loss = normalizer.sum() - (target*scores).sum() + 6*np.dot(w,w)
        return loss, (prob-target).ravel() @ matrix + 12*w
    sol = minimize(fg,np.zeros(matrix.shape[1]),jac=True,method="L-BFGS-B",
                   options={"maxiter":1000,"ftol":1e-9})
    if not sol.success:
        raise RuntimeError(sol.message)
    test = test - test.mean(axis=1, keepdims=True)
    return np.clip(test/scale,-6,6) @ sol.x


def standardized(v):
    return (v-v.mean(axis=1,keepdims=True)) / np.maximum(v.std(axis=1,keepdims=True),1e-6)


def combine(lin, tree, fraction):
    w = fraction/100
    return (1-w)*standardized(lin) + w*standardized(tree)


def fit_split(train, test, fractions=WEIGHTS):
    """Fit on development participants and score the complete feature set."""
    if {c.user for c in train} & {c.user for c in test}:
        raise ValueError("A training participant also occurs in evaluation")
    bank = base.Uptake(train)
    x = np.concatenate((base.concatenate(train, bank),
                        np.stack([c.coords for c in train]),
                        np.stack([c.personal for c in train])), axis=2)
    q = np.concatenate((base.concatenate(test, bank),
                        np.stack([c.coords for c in test]),
                        np.stack([c.personal for c in test])), axis=2)
    y = np.stack([c.y for c in train])
    linear = conditional_fit(x,y,q)
    tree = HistGradientBoostingClassifier(max_iter=180,learning_rate=.045,max_leaf_nodes=12,
                       min_samples_leaf=65,l2_regularization=10,random_state=SEED,early_stopping=False)
    tree.fit(x.reshape(-1,x.shape[-1]),y.ravel())
    t = logit(np.clip(tree.predict_proba(q.reshape(-1,q.shape[-1]))[:,1],1e-6,1-1e-6)).reshape(q.shape[:2])
    return {f"personal{weight}": combine(linear, t, weight) for weight in fractions}


def run(args):
    out, root = args.out, args.root
    protocol = out/"personal_protocol.json"; lock_file=out/"personal_lock.json"
    source = base.digest(Path(__file__))
    allusers = pd.read_parquet(root/"data/users.parquet",columns=["userId"]).userId.astype(int).tolist()
    dev = [u for u in allusers if fusion.participant_fold(u)==0]
    cases=assemble(root,args.movie_zip,dev)
    rows=[]
    for ix, jx in GroupKFold(4).split(cases,groups=[c.user for c in cases]):
        train=[cases[i] for i in ix]; test=[cases[i] for i in jx]
        rows.append(base.rows_for(test,fit_split(train,test)))
    frame=pd.concat(rows,ignore_index=True); score=base.report(frame)
    selected=max(WEIGHTS,key=lambda w:(score["metrics"][f"personal{w}"]["hit1"],
                                       score["metrics"][f"personal{w}"]["ndcg"], -w))
    write_json(lock_file,{"fraction":selected,"source_sha256":source,"protocol_sha256":base.digest(protocol),
                          "development":score})
    print("Frozen weight",selected,score["metrics"],flush=True)
    return

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--movie-zip",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=2): run(args)
