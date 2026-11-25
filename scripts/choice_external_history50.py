"""Refit the original static six-input MIND model using at most 50 history clicks."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from choice_external_mind import build_ranks, evaluate, fit, linear_scores, read_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--publisher-split", action="store_true")
    args = parser.parse_args()
    data = args.data
    paths = [data / name for name in ("MINDsmall_train.zip", "MINDsmall_dev.zip", "gain_full_large.zip")]
    with TemporaryDirectory(dir=data, prefix="history50-") as directory:
        capped_paths = []
        for path in paths:
            capped = Path(directory) / path.name
            with ZipFile(path) as src, ZipFile(capped, "w") as dst:
                dst.writestr("news.tsv", src.read("news.tsv"))
                with dst.open("behaviors.tsv", "w") as out:
                    for row in src.open("behaviors.tsv"):
                        values = row.split(b"\t", 4)
                        values[3] = b" ".join(values[3].split()[-50:])
                        out.write(b"\t".join(values))
            capped_paths.append(capped)
        periods, meta = read_inputs(capped_paths)
    train = periods[0]
    stem = "local50"
    holdout_digest = None
    if args.publisher_split:
        # NewsReX get_train_val_data uses DataFrame.sample(frac=1,random_state=42)
        # and reserves the first floor(5%*N) shuffled indices (7,848 impressions).
        order = pd.Series(np.arange(len(train[0]))).sample(frac=1, random_state=42).to_numpy()
        excluded = np.sort(order[:int(len(order) * .05)])
        keep = np.ones(len(order), dtype=bool)
        keep[excluded] = False
        selected_candidates = np.repeat(keep, np.diff(train[4]))
        offsets = np.r_[0, np.cumsum(np.diff(train[4])[keep])].astype(np.int32)
        holdout_digest = hashlib.sha256(train[0][excluded].astype("<i4").tobytes()).hexdigest()
        train = (*(values[keep] for values in train[:4]), offsets,
                 train[5][selected_candidates], train[6][selected_candidates])
        stem = "matched50"
    counts = {}
    for name, path in zip(("train", "small", "full"), paths):
        with ZipFile(path) as z:
            counts[name] = sum(len(row.split(b"\t", 4)[3].split()) > 50 for row in z.open("behaviors.tsv"))
    state = tuple(np.zeros(len(meta[0]), np.int64) for _ in range(3))
    x = build_ranks(train[2], train[3], train[4], train[5], train[6], *meta, *state, True)
    with threadpool_limits(limits=2):
        rule = fit(x, train, list(range(6)))
    del x
    output = {"history_limit_train_and_score": 50, "impressions_with_longer_input_before_split": counts,
              "train_impressions": len(train[0]), "held_out_ordered_ids_sha256": holdout_digest, "rule": rule}
    out = "ext_matched50_fit.json" if args.publisher_split else "ext_history50_fit.json"
    (data / out).write_text(json.dumps(output, indent=2) + "\n")
    for part, period in zip(("small", "full"), periods[1:]):
        features = build_ranks(period[2], period[3], period[4], period[5], period[6], *meta, *state, False)
        raw = linear_scores(features, np.array(rule["coef"]), np.array(rule["cols"], np.int32))
        _, ranks = evaluate(raw, period[0], period[4], period[5], period[6])
        np.savez_compressed(data / f"ext_{part}_{stem}.npz", ranks=ranks)
        print("Scored", part, len(period[0]), flush=True)


if __name__ == "__main__":
    main()
