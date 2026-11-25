"""Audit the original MIND-large article identities and attach a public-source manifest."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from choice_external_benchmark import digest, ROOT
from choice_external_mind import METRICS
from choice_neural_gain_stream import read


def article_urls(path):
    with ZipFile(path) as archive:
        pairs = [row.split(b"\t", 6) for row in archive.open("news.tsv")]
    return {int(row[0][1:]): row[5] for row in pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--full-scorer", metavar="MODEL",
                        help="Also stream all predictions through Microsoft's complete evaluator.")
    parser.add_argument("--verified-original", nargs=2, metavar=("SMALL_JSON", "FULL_JSON"), type=Path,
                        help="Attach completed original-evaluator outputs from choice_external_verify.py.")
    args = parser.parse_args()
    data = args.data
    results = json.loads(args.results.read_text())
    large = data / "MINDlarge_dev.zip"
    remapped = data / "gain_full_large.zip"
    original_urls = article_urls(large)
    remapped_urls = article_urls(remapped)
    old = read(large)
    new = read(remapped)
    # Different local IDs can represent the same public URL. Compare actual URLs in sequence.
    all_urls = {url: i for i, url in enumerate(set(remapped_urls.values()) | set(original_urls.values()))}
    def index(mapping):
        lookup = np.full(max(mapping)+1, -1, dtype=np.int32)
        for item, url in mapping.items():
            lookup[item] = all_urls[url]
        return lookup
    raw_lookup = index(original_urls)
    local_lookup = index(remapped_urls)
    np.testing.assert_array_equal(raw_lookup[old.items], local_lookup[new.items])
    for key in ("ids", "users", "hours", "offsets", "labels"):
        np.testing.assert_array_equal(getattr(old, key), getattr(new, key))
    releases = {
        "NRMS": "0f7f34ba5b6c0266304751fe54f7901085ca186c",
        "NAML": "a44ff655ebb58a5f6d956163f3c9be66eabc5169",
        "LSTUR": "44c54c7dc29a6bc6846015f614893e7f8a03ff04"}
    results["source_manifest"] = {
        "publisher": "https://msnews.github.io/",
        "evaluation": "complete public small and large validation, same logged day; separate screened sensitivity cohort",
        "training": "released MIND-small neural weights with random internal 5% validation and sampled training negatives; static local model uses all MIND-small training impressions; Switch additionally uses delayed MIND-small same-day evidence",
        "lt_user_transfer": "LSTUR uses publisher account embeddings on small; large transfer uses zero long-term state because there is no verified cross-release account-ID mapping",
        "software": "https://github.com/igor17400/NewsReX/tree/6f10b83a613be934a679aeae1cef595c39b636c0",
        "archives_sha256": {f.name: digest(f) for f in (data/"MINDsmall_train.zip", data/"MINDsmall_dev.zip", large)},
        "verified_ordered_large_candidate_urls": len(old.items),
        "weights": {},
        "ensemble_rules": {
                           "Neural-9": "equal mean over the three family ensemble scores",
                           "NoStream": "static four neural checkpoints and the six-input local model, original uncapped local profiles",
                           "NoStream50": "same fusion as NoStream, six-input model refitted and scored with at most 50 history clicks",
                           "Matched50": "same as NoStream50, local ranker and article counts only use the publisher's 149,117 fitting impressions",
                           "Switch": "stored frozen policy ranks, with delayed within-day MIND-small feedback"},
        "intervals": "normal 95% paired intervals clustered by public account; pointwise, conditional on fitted models and on the logged day",
    }
    for model, rev in releases.items():
        for seed in (42,123,456):
            path = data / ("nrms.safetensors" if (model, seed)==("NRMS", 123)
                           else f"{model}_{seed}_model.safetensors")
            results["source_manifest"]["weights"][f"{model}-{seed}"] = {
                "sha256": digest(path),
                "url": f"https://huggingface.co/newsrex/{model}-PYTORCH-MIND-small-glove-random/blob/{rev}/seed_{seed}/model.safetensors"}
    if args.full_scorer:
        model = args.full_scorer
        for part, cohort in (("small", "MINDsmall_dev.zip"), ("full", "MINDlarge_dev.zip")):
            target = results["cohorts"][part]["models"][model]
            with TemporaryDirectory(dir=data, prefix="official-") as folder:
                folder = Path(folder)
                prediction = folder / "ranks.npz"
                expected = folder / "expected.json"
                audit = folder / "verified.json"
                np.savez(prediction, ranks=np.load(data / f"ext_ranks_{part}_{model}.npy"))
                expected.write_text(json.dumps({"scores": {model: {k: target[k] for k in METRICS}}}))
                subprocess.run([sys.executable, str(ROOT/"scripts/choice_external_verify.py"),
                                "--data", str(data), "--cohort", str(data/cohort), "--evaluator", str(data/"evaluate.py"),
                                "--ranks", str(prediction), "--model", model, "--table", "scores",
                                "--results", str(expected), "--output", str(audit)], check=True)
                target["unmodified_end_to_end_all_lists"] = json.loads(audit.read_text())
    if args.verified_original:
        for part, source in zip(("small", "full"), args.verified_original):
            proof = json.loads(source.read_text())
            cohort = results["cohorts"][part]
            target = cohort["models"][proof["model"]]
            if (proof["validation_impressions"] != cohort["impressions"] or
                    proof["validation_candidates"] != cohort["candidates"] or
                    proof["scorer_sha256"] != results["source_sha256"] or
                    max(abs(target[k] - proof["official_scores"][k]) for k in METRICS) > 1e-10):
                raise ValueError("Original-evaluator proof does not match the current ranking metrics")
            target["unmodified_end_to_end_all_lists"] = proof
    args.results.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
