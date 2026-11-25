"""Score saved MIND predictions using the unmodified official evaluator, offline."""

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://raw.githubusercontent.com/msnews/MIND/47fb9852c97814d80ccf9c658d6b81f5c930b510/evaluate.py"
EXPECTED = "c6195008ed19ba7e68145069cffbdbcc0582ff0e983e2a5966095274043cc2ca"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--table", default="models")
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--ranks", type=Path, required=True)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    if hashlib.sha256(args.evaluator.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("Official evaluator source digest mismatch")
    spec = importlib.util.spec_from_file_location(
        "official_mind_scorer", args.evaluator
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ranks = np.load(args.ranks)["ranks"]
    truth, submission = io.StringIO(), io.StringIO()
    start = 0
    count = 0
    with ZipFile(args.cohort or args.data / "MINDsmall_dev.zip") as z:
        for row in z.open("behaviors.tsv"):
            sid, *_, slate = row.rstrip(b"\n").split(b"\t")
            ys = [int(token[-1:]) for token in slate.split()]
            end = start + len(ys)
            predicted = ranks[start:end]
            if not np.array_equal(np.sort(predicted), np.arange(1, end - start + 1)):
                raise ValueError("Invalid rank permutation")
            truth.write(
                sid.decode() + " " + json.dumps(ys, separators=(",", ":")) + "\n"
            )
            submission.write(
                sid.decode()
                + " "
                + json.dumps(predicted.tolist(), separators=(",", ":"))
                + "\n"
            )
            start = end
            count += 1
    if start != len(ranks):
        raise ValueError("Prediction count mismatch")
    truth.seek(0)
    submission.seek(0)
    values = dict(
        zip(("AUC", "MRR", "NDCG@5", "NDCG@10"), module.scoring(truth, submission))
    )
    local = json.loads(args.results.read_text())[args.table][args.model]
    err = max(abs(values[k] - local[k]) for k in local)
    if err > 1e-10:
        raise ValueError(f"Official scorer discrepancy: {err}")
    output = dict(
        source=SOURCE,
        scorer_sha256=EXPECTED,
        validation_impressions=count,
        validation_candidates=start,
        model=args.model,
        official_scores=values,
        max_abs_difference=err,
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
