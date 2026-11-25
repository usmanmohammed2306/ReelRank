"""Published rank features and paired evaluation utilities for SM-RS."""
import hashlib

import numpy as np
import pandas as pd


EXPERTS = ("popularity", "ease", "implicit_mf", "user_knn", "std_vae", "mlt_vae")
METRICS = ("ndcg", "p5", "hit1", "hit5", "first")
DRAWS = 5000


def participant_fold(user):
    key = hashlib.sha256(f"reelrank-choice-v1:{int(user)}".encode()).digest()
    return int.from_bytes(key[:8], "big") % 5


def candidate_ranks(candidates, orders):
    ids = list(map(int, candidates))
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError("A recommendation slate must contain ten unique alternatives.")
    ix = {item: n for n, item in enumerate(ids)}
    ranks = np.empty((len(ids), len(orders)))
    for k, order in enumerate(orders):
        if len(order) != len(ids) or set(order) != set(ids):
            raise ValueError("A source ranking does not match its displayed alternatives.")
        ranks[[ix[int(item)] for item in order], k] = np.linspace(1.0, 0.0, len(ids))
    return ranks


def design(r, domain, history_size, cumulative, recent):
    base = np.column_stack([r, r*r, r == 1.0])
    # +/- coding permits separate domain effects and shrinks them toward a
    # common estimate. No component uses the original displayed position.
    static = np.column_stack([base, base if domain == "books" else -base])
    spread = np.std(r,axis=1,keepdims=True)
    consensus = r.mean(axis=1,keepdims=True)
    context = np.column_stack([r*spread, r*spread.mean(), r*consensus,
                               r*np.log1p(history_size)/4, spread, consensus**2,
                               r.max(axis=1), r.min(axis=1)])
    personal = np.column_stack([4*r*cumulative, 4*r*recent, 4*r*r*cumulative])
    features = np.column_stack([static,context,personal])
    return features, (static.shape[1], static.shape[1]+context.shape[1], features.shape[1])


def read_orders(root):
    # All rankers must cover exactly the same users and slates.
    rows = {}
    expected = None
    for method in EXPERTS:
        data = pd.read_csv(root/f"recommendation_{method}_ranks.csv.gz")
        if data.duplicated(["userId","impressionId"]).any():
            raise ValueError(f"Duplicate ranking in {method}")
        rank = {(r.userId,r.impressionId): [int(t) for t in r.ranking.split(",")] for r in data.itertuples()}
        if expected is not None and rank.keys()!=expected:
            raise ValueError("Expert coverage differs across recommendation slates")
        expected = rank.keys()
        rows[method] = rank
    return rows


def tie_key(ids, slate_id):
    keys = np.asarray(ids, dtype=np.uint64) ^ np.uint64((slate_id+1)*1000003)
    keys ^= keys >> np.uint64(30)
    keys *= np.uint64(0xbf58476d1ce4e5b9)
    keys ^= keys >> np.uint64(27)
    keys *= np.uint64(0x94d049bb133111eb)
    return keys ^ (keys >> np.uint64(31))


def intervals(rows,adaptive="adaptive"):
    metric = ["ndcg","p5","hit1","hit5","first"]
    own = rows[rows.method==adaptive].set_index(["user","slate"])[metric].sort_index()
    users = own.index.get_level_values("user")
    user_ids = users.unique()
    counts = own.groupby(level=0).size().reindex(user_ids).to_numpy()
    rng = np.random.default_rng(57517)
    draws = rng.multinomial(len(user_ids),np.full(len(user_ids),1/len(user_ids)),size=DRAWS)
    denominator = draws @ counts
    results = {}
    for baseline in sorted(set(rows.method)-{adaptive}):
        other = rows[rows.method==baseline].set_index(["user","slate"])[metric].reindex(own.index)
        if other.isna().any().any():
            raise ValueError("Paired comparison has an incomplete slate.")
        diff = own - other
        sums = diff.groupby(level=0).sum().reindex(user_ids).to_numpy()
        draws_mean = (draws @ sums)/denominator[:,None]
        low,high = np.quantile(draws_mean,[.025,.975],axis=0)
        results[baseline] = {m:{"difference":float(diff[m].mean()),"ci95":[float(low[k]),float(high[k])]}
                             for k,m in enumerate(metric)}
    return results
