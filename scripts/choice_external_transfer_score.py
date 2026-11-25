"""Apply the learned MIND-small rank coefficients to a URL-reindexed large cohort."""

import argparse
import json
from pathlib import Path

import numpy as np

from choice_external_mind import (
    ROOT,
    build_ranks,
    evaluate,
    linear_scores,
    read_inputs,
    source_hash,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--private-ranks", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(
        (ROOT / "results/choice_gains/mind_inputs.json").read_text()
    )
    train_path = args.data / "MINDsmall_train.zip"
    if source_hash(train_path) != protocol["sha256_train"]:
        raise ValueError("MIND-small training digest mismatch")
    periods, meta = read_inputs([train_path, args.cohort])
    train, dev = periods
    n = len(meta[0])
    sizes = np.diff(train[4])
    last = np.repeat(train[2] == train[2].max(), sizes)
    clicks = np.bincount(train[5], weights=train[6], minlength=n).astype(np.int64)
    shows = np.bincount(train[5], minlength=n).astype(np.int64)
    recent = np.bincount(train[5][last], weights=train[6][last], minlength=n).astype(
        np.int64
    )
    x = build_ranks(
        dev[2], dev[3], dev[4], dev[5], dev[6], *meta, clicks, shows, recent, False
    )
    rule = protocol["local_six_signal_model"]
    scores = linear_scores(x, np.array(rule["coef"]), np.array(rule["cols"], np.int32))
    _, ranks = evaluate(scores, dev[0], dev[4], dev[5], dev[6])
    np.savez_compressed(args.private_ranks, ranks=ranks)


if __name__ == "__main__":
    main()
