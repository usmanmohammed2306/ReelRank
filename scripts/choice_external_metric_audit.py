"""Rescore every MIND impression with the publisher's four original metric functions.

With strict integer ranks, different lists can have identical metric inputs. Cache
only mathematically equivalent cases. The benchmark script separately calls the
original end-to-end scoring() function on a random sample of actual input lines.
"""
import argparse
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from choice_external_benchmark import digest
from choice_external_mind import METRICS
from choice_external_verify import EXPECTED
from choice_neural_gain_stream import read


def original_functions(scorer):
    @lru_cache(None)
    def auc(n, p, rank_sum):
        # AUC averages every positive-negative comparison. For p positives in
        # distinct positions, the inversion count is sum(ranks) - p*(p+1)/2.
        positions = np.arange(p)
        remaining = rank_sum - p*(p+1)//2
        q, r = divmod(remaining, n-p)
        if q:
            positions[-q:] += n-p
        if r:
            positions[p-q-1] += r
        labels = np.zeros(n)
        labels[positions] = 1
        return scorer.roc_auc_score(labels, 1./np.arange(1, n+1))

    @lru_cache(None)
    def mrr(positions):
        # Adding irrelevant items after the lowest positive does not affect MRR.
        labels = np.zeros(positions[-1])
        labels[np.array(positions)-1] = 1
        return scorer.mrr_score(labels, 1./np.arange(1, len(labels)+1))

    @lru_cache(None)
    def ndcg(k, ideal_positives, top_positions):
        # Only relevance at ranks <=k and the capped ideal positive count matter.
        labels = np.zeros(k+ideal_positives)
        labels[np.array(top_positions, dtype=int)-1] = 1
        labels[k:k+ideal_positives-len(top_positions)] = 1
        return scorer.ndcg_score(labels, 1./np.arange(1, len(labels)+1), k)

    return auc, mrr, ndcg


def evaluate_functions(ranks, labels, offsets, functions):
    auc, mrr, ndcg = functions
    pos = ranks[labels.astype(bool)]
    starts = np.r_[0, np.cumsum(np.add.reduceat(labels.astype(np.int32), offsets[:-1]))]
    result = np.empty((len(offsets)-1, 4))
    for i, n in enumerate(np.diff(offsets)):
        positions = tuple(sorted(pos[starts[i]:starts[i+1]].tolist()))
        p = len(positions)
        top10 = tuple(j for j in positions if j <= 10)
        top5 = tuple(j for j in top10 if j <= 5)
        result[i] = (auc(int(n), p, sum(positions)), mrr(positions),
                     ndcg(5, min(5,p), top5), ndcg(10, min(10,p), top10))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluator = args.data / "evaluate.py"
    if digest(evaluator) != EXPECTED:
        raise ValueError("MIND evaluator hash mismatch")
    spec = importlib.util.spec_from_file_location("original_mind_scorer", evaluator)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    functions = original_functions(scorer)
    results = json.loads(args.results.read_text())
    for part, filename in [("small", "MINDsmall_dev.zip"), ("full", "MINDlarge_dev.zip")]:
        period = read(args.data / filename)
        masks = {part: np.ones(len(period.ids), bool)}
        if part == "full":
            with ZipFile(args.data / "large_cohort.zip") as z:
                ids = [int(row.split(b'\t', 1)[0]) for row in z.open("behaviors.tsv")]
            masks["screened"] = np.isin(period.ids, ids)
        for name in results["cohorts"][part]["models"]:
            ranks = np.load(args.data / f"ext_ranks_{part}_{name}.npy", mmap_mode="r")
            values = evaluate_functions(ranks, period.labels, period.offsets, functions)
            for key, mask in masks.items():
                model = results["cohorts"][key]["models"][name]
                official = values[mask].mean(axis=0)
                err = float(max(abs(model[k]-v) for k,v in zip(METRICS, official)))
                if err > 1e-10:
                    raise ValueError(f"Complete original metric discrepancy: {part}/{name}, {err}")
                model["original_metric_functions_all_lists"] = {
                    "impressions": int(mask.sum()), "max_abs_error": err,
                    "scores": dict(zip(METRICS, official))}
            print("Original metrics verified:", part, name, flush=True)
    results["original_metric_evaluations"] = {
        name: f.cache_info().misses for name, f in zip(("AUC", "MRR", "nDCG"), functions)}
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
