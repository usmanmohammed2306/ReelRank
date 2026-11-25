"""Feature construction, fitting and evaluation for the MIND recommenders."""

import hashlib
from array import array
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from numba import njit
from scipy.optimize import minimize
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["AUC", "MRR", "NDCG@5", "NDCG@10"]


def read_inputs(paths):
    articles = {}
    topics = [{}, {}]
    histories = {b"": 0}
    arrays = []
    for path in paths:
        with ZipFile(path) as z:
            for line in z.open("news.tsv"):
                article, category, subcategory, title = line.split(b"\t", 4)[:4]
                identifier = int(article[1:])
                if identifier not in articles:
                    categories = [
                        mapping.setdefault(value, len(mapping))
                        for mapping, value in zip(topics, (category, subcategory))
                    ]
                    articles[identifier] = (*categories, title.decode("utf8"))
            ids, days, users, profile_ids, items, labels, offsets = (
                [],
                [],
                [],
                [],
                array("i"),
                bytearray(),
                [0],
            )
            for line in z.open("behaviors.tsv"):
                sid, user, date, history, slate = line.rstrip(b"\r\n").split(b"\t")
                ids.append(int(sid))
                users.append(int(user[1:]))
                # The official small release covers November 2019 only.
                days.append(int(date.split(b"/")[1]))
                profile_ids.append(histories.setdefault(history, len(histories)))
                for token in slate.split():
                    items.append(int(token[1:-2]))
                    labels.append(int(token[-1:]))
                offsets.append(len(items))
            arrays.append(
                tuple(
                    np.asarray(v, dtype=np.int32)
                    for v in (ids, users, days, profile_ids, offsets)
                )
                + (
                    np.frombuffer(items, dtype=np.int32),
                    np.frombuffer(labels, dtype=np.uint8),
                )
            )
    identifiers = np.array(list(articles))
    article_topics = np.zeros((identifiers.max() + 1, 2), dtype=np.int32)
    known = np.zeros(len(article_topics), dtype=bool)
    known[identifiers] = True
    article_topics[identifiers] = [row[:2] for row in articles.values()]
    hasher = HashingVectorizer(
        n_features=256, dtype=np.float32, norm="l2", alternate_sign=True
    )
    embeddings = np.zeros((len(article_topics), 256), dtype=np.float32)
    embeddings[identifiers] = hasher.transform(
        row[2] for row in articles.values()
    ).toarray()
    category_history = np.zeros((len(histories), len(topics[0])), dtype=np.int32)
    subcategory_history = np.zeros((len(histories), len(topics[1])), dtype=np.int32)
    title_history = np.zeros((len(histories), 256), dtype=np.float32)
    for history, index in histories.items():
        if not history:
            continue
        past = np.array([int(token[1:]) for token in history.split()], dtype=np.int32)
        if past.max() >= len(known) or not known[past].all():
            raise ValueError("Missing historical article metadata")
        np.add.at(category_history[index], article_topics[past, 0], 1)
        np.add.at(subcategory_history[index], article_topics[past, 1], 1)
        center = embeddings[past].sum(axis=0)
        title_history[index] = center / max(np.linalg.norm(center), 1e-12)
    return arrays, (
        article_topics,
        embeddings,
        category_history,
        subcategory_history,
        title_history,
    )


@njit
def to_percentile(scores):
    n = len(scores)
    order = np.argsort(scores)
    ranks = np.zeros(n, dtype=np.float32)
    lo = 0
    while lo < n:
        hi = lo + 1
        while hi < n and scores[order[hi]] == scores[order[lo]]:
            hi += 1
        mid = (lo + hi - 1) / (2 * max(1, n - 1))
        for j in range(lo, hi):
            ranks[order[j]] = mid
        lo = hi
    return ranks


@njit
def build_ranks(
    days,
    histories,
    offsets,
    items,
    labels,
    article_topics,
    embeddings,
    category_history,
    subcategory_history,
    title_history,
    clicks,
    shows,
    recent,
    update,
):
    features = np.empty((len(items), 6), dtype=np.float32)
    for day in np.unique(days):
        next_recent = np.zeros_like(clicks)
        new_shows = np.zeros_like(shows)
        prior = clicks.sum() / max(1, shows.sum())
        for row in range(len(days)):
            if days[row] != day:
                continue
            a, b = offsets[row : row + 2]
            h = histories[row]
            raw = np.empty((b - a, 6), dtype=np.float64)
            for j in range(a, b):
                item = items[j]
                raw[j - a, 0] = clicks[item]
                raw[j - a, 1] = (clicks[item] + 20 * prior) / (shows[item] + 20)
                raw[j - a, 2] = recent[item]
                raw[j - a, 3] = category_history[h, article_topics[item, 0]]
                raw[j - a, 4] = subcategory_history[h, article_topics[item, 1]]
                raw[j - a, 5] = np.dot(embeddings[item], title_history[h])
                if update:
                    new_shows[item] += 1
                    next_recent[item] += labels[j]
            for col in range(6):
                features[a:b, col] = to_percentile(raw[:, col])
        if update:
            clicks += next_recent
            shows += new_shows
            recent[:] = next_recent
    return features


@njit
def linear_scores(x, beta, cols):
    scores = np.zeros(len(x), dtype=np.float64)
    d = len(cols)
    for i in range(len(x)):
        for j in range(d):
            r = x[i, cols[j]]
            scores[i] += (
                r * beta[j] + r * r * beta[d + j] + (r >= 0.75) * beta[2 * d + j]
            )
    return scores


@njit
def objective(beta, x, y, offsets, cols):
    scores = linear_scores(x, beta, cols)
    grad = np.zeros_like(beta)
    loss = 0.0
    d = len(cols)
    num = len(offsets) - 1
    for a, b in zip(offsets[:-1], offsets[1:]):
        target = y[a:b].sum()
        max_score = np.max(scores[a:b])
        normalizer = 0.0
        observed = 0.0
        for i in range(a, b):
            observed += y[i] * scores[i]
            scores[i] = np.exp(scores[i] - max_score)
            normalizer += scores[i]
        loss += np.log(normalizer) + max_score - observed / target
        for i in range(a, b):
            residual = scores[i] / normalizer - y[i] / target
            for j in range(d):
                r = x[i, cols[j]]
                grad[j] += residual * r
                grad[j + d] += residual * r * r
                grad[j + 2 * d] += residual * (r >= 0.75)
    return loss / num + 0.0005 * np.dot(beta, beta), grad / num + 0.001 * beta


def fit(x, period, cols):
    offsets, y = period[4], period[6]
    if (np.add.reduceat(y.astype(np.int32), offsets[:-1]) == 0).any():
        raise ValueError("A training impression has zero clicks.")
    cols = np.asarray(cols, dtype=np.int32)
    result = minimize(
        objective,
        np.zeros(3 * len(cols)),
        args=(x, y, offsets, cols),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 150, "ftol": 1e-11, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return {
        "cols": cols.tolist(),
        "coef": result.x.tolist(),
        "iterations": int(result.nit),
    }


@njit
def tie_hash(sid, item):
    # Bijective 32-bit integer mixing for fixed impression. Independent of TSV ordering.
    v = (item ^ ((sid * 2654435761) & 0xFFFFFFFF)) & 0xFFFFFFFF
    v = ((v ^ (v >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    v = ((v ^ (v >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return (v ^ (v >> 16)) & 0xFFFFFFFF


@njit
def evaluate(scores, ids, offsets, items, labels):
    metrics = np.empty((len(ids), 4), np.float64)
    ranks = np.empty(len(labels), dtype=np.int32)
    for k in range(len(ids)):
        a, b = offsets[k : k + 2]
        n = b - a
        p = labels[a:b].sum()
        if p <= 0 or p >= n:
            raise ValueError("An impression lacks both click classes")
        hashes = np.array([tie_hash(ids[k], item) for item in items[a:b]])
        independent_order = np.argsort(hashes)
        order = independent_order[
            np.argsort(-scores[a:b][independent_order], kind="mergesort")
        ]
        rr, auc, nd5, nd10, id5, id10, positives = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for j, index in enumerate(order):
            ranks[a + index] = j + 1
            weight = 1 / np.log2(j + 2)
            if j < p and j < 10:
                id10 += weight
                if j < 5:
                    id5 += weight
            if labels[a + index]:
                positives += 1
                rr += 1.0 / (j + 1)
                if j < 10:
                    nd10 += weight
                    if j < 5:
                        nd5 += weight
            else:
                auc += positives
        metrics[k] = np.array([auc / (p * (n - p)), rr / p, nd5 / id5, nd10 / id10])
    return metrics, ranks


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
