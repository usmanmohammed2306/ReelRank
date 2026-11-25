"""Independently grade the frozen switch and published MIND checkpoints on raw click labels."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from numba import njit

from choice_external_mind import evaluate, METRICS
from choice_external_verify import EXPECTED, SOURCE
from choice_neural_gain import rank_scores
from choice_neural_gain_stream import read

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 456)
METRIC_NAMES = (*METRICS, "Hit@1")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@njit
def grade(labels, offsets, ranks):
    """Grade by reading positive ranks directly, independent of the score-order evaluator."""
    output = np.zeros((len(offsets)-1, 5))
    for i in range(len(output)):
        lo, hi = offsets[i:i + 2]
        n = hi - lo
        positive = 0
        rank_sum = 0
        seen = np.zeros(n, np.uint8)
        for j in range(lo, hi):
            rank = ranks[j]
            if rank < 1 or rank > n or seen[rank - 1]:
                raise ValueError("Invalid rank permutation")
            seen[rank - 1] = 1
            if labels[j]:
                positive += 1
                rank_sum += rank
                output[i, 1] += 1 / rank
                if rank <= 10:
                    output[i, 3] += 1 / np.log2(1 + rank)
                if rank <= 5:
                    output[i, 2] += 1 / np.log2(1 + rank)
                if rank == 1:
                    output[i, 4] = 1
        if positive == 0 or positive == n:
            raise ValueError("Undefined two-class impression")
        output[i, 0] = (positive * n - rank_sum - positive * (positive - 1) / 2) / (positive * (n - positive))
        output[i, 1] /= positive
        for col, cutoff in ((2, 5), (3, 10)):
            ideal = 0.
            for rank in range(1, min(positive, cutoff) + 1):
                ideal += 1 / np.log2(1 + rank)
            output[i, col] /= ideal
    return output


def ci(delta, groups):
    mean = delta.mean(axis=0)
    g = groups.max() + 1
    residual = np.zeros((g, delta.shape[1]))
    np.add.at(residual, groups, delta - mean)
    margin = 1.95996398454 * np.sqrt(g / (g-1) * np.sum(residual**2, axis=0)) / len(delta)
    return {k: [float(d), float(d-m), float(d+m)] for k, d, m in zip(METRIC_NAMES, mean, margin)}


def official_check(scorer, ids, offsets, labels, ranks, idx, metric):
    truth, prediction = io.StringIO(), io.StringIO()
    for i in idx:
        lo, hi = offsets[i:i+2]
        truth.write(f"{ids[i]} {json.dumps(labels[lo:hi].tolist(), separators=(',', ':'))}\n")
        prediction.write(f"{ids[i]} {json.dumps(ranks[lo:hi].tolist(), separators=(',', ':'))}\n")
    truth.seek(0)
    prediction.seek(0)
    official = scorer.scoring(truth, prediction)
    error = float(np.abs(np.array(official)-metric[idx, :4].mean(axis=0)).max())
    if error > 1e-10:
        raise ValueError(f"Official evaluator discrepancy: {error}")
    return {"impressions": len(idx), "max_abs_error": error,
            "scores": dict(zip(METRICS, official))}


def generate(data, cohort, part, include_lstur, include_matched):
    def load(name):
        return np.load(data / name, mmap_mode="r")
    naml = np.zeros(len(cohort.labels), np.float32)
    nrms = np.zeros_like(naml)
    for family in ("NAML", "NRMS"):
        for seed in SEEDS:
            if family == "NAML":
                raw = load(f"gain_{part}_naml_{seed}.npy")
                # Use cached encoder outputs in ReelRank and the ensembles.
                naml += rank_scores(load(f"gain_{part}_naml_{seed}_cached.npy"), cohort.offsets) / 3
            elif seed == 123:
                raw = load("nrms_scores.npy" if part == "small" else "gain_full_nrms.npy")
                nrms123 = rank_scores(raw, cohort.offsets)
                nrms += nrms123 / 3
            else:
                raw = load(f"gain_{part}_nrms_{seed}.npy")
                nrms += rank_scores(raw, cohort.offsets) / 3
            if not np.isfinite(raw).all() or len(raw) != len(naml):
                raise ValueError("Missing or nonfinite checkpoint scores")
            yield f"{family}-{seed}", raw, False
    if include_lstur:
        lstur = np.zeros_like(naml)
        for seed in SEEDS:
            raw = load(f"gain_{part}_lstur_{seed}.npy")
            if not np.isfinite(raw).all() or len(raw) != len(naml):
                raise ValueError("Missing or nonfinite LSTUR scores")
            lstur += rank_scores(raw, cohort.offsets) / 3
            yield f"LSTUR-{seed}", raw, False
        yield "Neural-9", (naml + nrms + lstur) / 3, False
    localfile = "full_ranks.npz" if part == "small" else "gain_full_local.npz"
    localrank = np.load(data / localfile)["ranks"]
    static = (.35*naml + .15*nrms123 + .2*rank_scores(-localrank, cohort.offsets)) / .7
    yield "NoStream", static, False
    if include_matched:
        capped = np.load(data / f"ext_{part}_local50.npz")["ranks"]
        yield "NoStream50", (.35*naml + .15*nrms123 + .2*rank_scores(-capped, cohort.offsets)) / .7, False
        matched = np.load(data / f"ext_{part}_matched50.npz")["ranks"]
        yield "Matched50", (.35*naml + .15*nrms123 + .2*rank_scores(-matched, cohort.offsets)) / .7, False
    yield "Switch", np.load(data / f"gain_{part}_Switch_ranks.npz")["ranks"], True


def run(args, part, scorer):
    data = args.data
    rawpath = data / ("MINDsmall_dev.zip" if part == "small" else "MINDlarge_dev.zip")
    raw = read(rawpath)
    expected = json.loads((ROOT / "results/choice_gains/mind_inputs.json").read_text())
    if digest(rawpath) != expected["sha256_dev" if part == "small" else "raw_sha256"]:
        raise ValueError("Incorrect publisher archive")
    canonical = raw if part == "small" else read(data / "gain_full_large.zip")
    for name in ("ids", "users", "labels", "offsets", "hours"):
        np.testing.assert_array_equal(getattr(raw, name), getattr(canonical, name))
    masks = {part: np.ones(len(raw.ids), bool)}
    if part == "full":
        with ZipFile(data / "large_cohort.zip") as z:
            screened_ids = [int(r.split(b'\t', 1)[0]) for r in z.open("behaviors.tsv")]
        masks["screened"] = np.isin(raw.ids, screened_ids)
        np.testing.assert_array_equal(raw.ids[masks["screened"]], screened_ids)
    reports = {}
    for name, mask in masks.items():
        reports[name] = {"impressions": int(mask.sum()), "accounts": len(np.unique(raw.users[mask])),
                         "candidates": int(np.diff(raw.offsets)[mask].sum()), "models": {}, "paired": {}}
    all_metrics = {}
    rng = np.random.default_rng(575)
    official_sample = np.sort(rng.choice(len(raw.ids), min(512, len(raw.ids)), replace=False))
    for name, value, ready in generate(data, canonical, part, args.lstur, args.history50):
        if ready:
            ranks = value
        else:
            _, ranks = evaluate(value, canonical.ids, canonical.offsets, canonical.items, canonical.labels)
        table = grade(raw.labels, raw.offsets, ranks)
        all_metrics[name] = table
        for key, mask in masks.items():
            selected = table[mask]
            reports[key]["models"][name] = dict(zip(METRIC_NAMES, selected.mean(axis=0)))
            reports[key]["models"][name]["first_hits"] = int(selected[:, 4].sum())
        reports[part]["models"][name]["original_scorer_audit"] = official_check(
            scorer, raw.ids, raw.offsets, raw.labels, ranks, official_sample, table)
        if args.save_ranks:
            np.save(data / f"ext_ranks_{part}_{name}.npy", ranks)
            if name in ("NoStream", "NoStream50", "Matched50"):
                np.save(data / f"ext_scores_{part}_{name}.npy", value)
        print(part, name, reports[part]["models"][name]["first_hits"], flush=True)
    for key, mask in masks.items():
        _, group = np.unique(raw.users[mask], return_inverse=True)
        targets = ("NoStream", "NoStream50", "Matched50", "Switch") if args.history50 else ("NoStream", "Switch")
        for target in targets:
            reports[key]["paired"][target] = {}
            for name, base in all_metrics.items():
                if name != target:
                    diff = all_metrics[target][mask] - base[mask]
                    reports[key]["paired"][target][name] = {
                        "extra_first_hits": int(diff[:, 4].sum()), "clustered_95": ci(diff, group)}
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--lstur", action="store_true")
    parser.add_argument("--history50", action="store_true")
    parser.add_argument("--save-ranks", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = args.data / "evaluate.py"
    if digest(path) != EXPECTED:
        raise ValueError("Official scorer digest mismatch")
    spec = importlib.util.spec_from_file_location("official_mind", path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    output = {"source": SOURCE, "source_sha256": EXPECTED, "cohorts": {}}
    for part in ("small", "full"):
        output["cohorts"].update(run(args, part, scorer))
    source = {}
    models = [(m, s) for m in ("NRMS", "NAML") for s in SEEDS]
    if args.lstur:
        models.extend(("LSTUR", seed) for seed in SEEDS)
    for model, seed in models:
        name = f"{model}-{seed}"
        path = args.data / f"{model}_{seed}_test_results.json"
        if name == "NRMS-123":
            path = args.data / "nrms_test.json"
        if model == "LSTUR":
            record = json.loads((args.data / f"LSTUR_{seed}_summary.json").read_text())["final_test_metrics"]
        else:
            record = json.loads(path.read_text())
        if record["_num_impressions"] != output["cohorts"]["small"]["impressions"]:
            raise ValueError("Publisher test size mismatch")
        reproduced = output["cohorts"]["small"]["models"][name]
        source[name] = {"published": {k: record[k.lower()] for k in METRICS},
                        "max_abs_local_difference": max(abs(reproduced[k]-record[k.lower()]) for k in METRICS)}
    output["publisher_small_validation"] = source
    if args.history50:
        output["history50_fit"] = json.loads((args.data / "ext_history50_fit.json").read_text())
        output["matched50_fit"] = json.loads((args.data / "ext_matched50_fit.json").read_text())
    args.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
