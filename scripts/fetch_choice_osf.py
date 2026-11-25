"""Obtain publisher-verified Task 3 artifacts for both SM-RS domains."""
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

MANIFEST = json.loads(Path(__file__).with_name("choice_osf_files.json").read_text())
DATA_COMMIT = "3c8af3b828d9a194d15f1b19abb739319cdb1751"
CORE_HASHES = {
    "behaviors": "2c757a8c30c434870a97d98235cc89e26b3f0becaddcb53ed4b6e1154e0cc34f",
    "books": "c56a86616ec4dd78fc6bce133a11a65df1ea1fdff4e002f84fccb50ec5d69f85",
    "comparative_diversity": "5e4e062dbc8e610394791d2bf6df3c9d495006a5c3212e0a74a2def6ded7b139",
    "criteria_values": "81c2b9479a59f76a621db370ebc15bf13ca0c40d5f63f96a4ef9f1b9fad75899",
    "movies": "41cd0b43efa3323b80f69a797f0bbb6ece3e9d964c54504ff4bfe91871429f01",
    "objective_perceptions": "4c54b41b5d39986ca2ade59279afb71534888e61aa714b06692fccce0897baca",
    "propensities": "1bcddd7b14ad550fda9cd75495b8b5627e4e62448f62bdcfad03f3dbb444afc7",
    "users": "0a82af9b7fa93cb3e016a46ae3fd84dcc1b054fe128b0dfd500d5b277859c8f5",
}
CORE = {
    name+".parquet": {"url":f"https://huggingface.co/datasets/pdokoupil/SM-RS/resolve/{DATA_COMMIT}/{name}.parquet",
                      "sha256":checksum} for name,checksum in CORE_HASHES.items()
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as src:
        for b in iter(lambda: src.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def fetch(dest, name, manifest=MANIFEST):
    path = dest / name
    record = manifest[name]
    if path.exists() and digest(path) == record["sha256"]:
        return
    with tempfile.TemporaryDirectory(dir=dest) as temp:
        tmp = Path(temp) / "download"
        subprocess.run(["curl", "-fsSL", "--retry", "4", "--retry-all-errors",
                        "--output", str(tmp), record["url"]], check=True)
        if digest(tmp) != record["sha256"]:
            raise ValueError(f"Checksum mismatch for {name}")
        os.replace(tmp, path)
    print(name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--core", action="store_true",
                        help="Fetch the complete pinned Hugging Face release into DIRECTORY")
    args = parser.parse_args()
    args.directory.mkdir(exist_ok=True, parents=True)
    manifest = CORE if args.core else MANIFEST
    files = args.names or list(manifest)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda name: fetch(args.directory, name, manifest), files))
