"""Infer the pinned NewsReX NRMS model on MIND-small, caching exact news vectors."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]


class CachedArticles(torch.nn.Module):
    def __init__(self, vectors):
        super().__init__()
        self.register_buffer("vectors", vectors)

    @staticmethod
    def valid_mask(ids):
        return ids != 0

    def forward(self, ids, training=False):
        return self.vectors[ids]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 123, 456), default=123)
    parser.add_argument(
        "--cohort", type=Path, help="Optional URL-reindexed large confirmation cohort."
    )
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    protocol = json.loads(
        (ROOT / "results/choice_gains/mind_inputs.json").read_text()
    )
    digests = {
        42: "ff0e9c16e777f2cfdf9ed37f07258677d37d947d4771a41e6561debc5c95ef07",
        123: protocol["nrms_seed_123_sha256"],
        456: "cbae60634035de12ef6075413b3c5c592e06767b3d44df26dbd21d75094affa7",
    }
    if hashlib.sha256(args.weights.read_bytes()).hexdigest() != digests[args.seed]:
        raise ValueError("NRMS checkpoint digest mismatch")
    sys.path.insert(0, str(args.upstream.resolve()))
    from src.core.data.processing.text.news import read_all_news
    from src.core.data.processing.text.vocabulary import build_vocabulary, tokenize_text
    from src.frameworks.pytorch.layers import GloveTextEncoder
    from src.frameworks.pytorch.models.nrms import NewsEncoder, UserEncoder, NRMSConfig

    for part, target in (("train", "train"), ("dev", "test")):
        with ZipFile(args.data / f"MINDsmall_{part}.zip") as z:
            z.extract("news.tsv", args.data / target)
    news = read_all_news(args.data)
    vocab = build_vocabulary(news, word_threshold=5)
    weights = load_file(str(args.weights))
    n, d = weights["news_encoder.text_encoder.embedding.weight"].shape
    if len(vocab) != n:
        raise ValueError(
            f"Checkpoint requires {n} vocabulary entries, reconstructed {len(vocab)}"
        )
    print(f"Vocabulary matches checkpoint: {n}", flush=True)
    cohort = args.cohort or args.data / "MINDsmall_dev.zip"
    titles = {int(nid[1:]): str(title) for nid, title in zip(news.id, news.title)}
    if args.cohort:
        with ZipFile(args.cohort) as z:
            for line in z.open("news.tsv"):
                fields = line.split(b"\t", 4)
                titles.setdefault(int(fields[0][1:]), fields[3].decode("utf8"))
    ids = np.array(list(titles), dtype=np.int64)
    tokens = np.zeros((ids.max() + 1, 32), dtype=np.int64)
    for item, title in titles.items():
        tokens[item] = tokenize_text(str(title), vocab, 32, 1, 0)
    cfg = NRMSConfig(
        news_num_heads=20, user_num_heads=20, head_dim=15, max_title_length=32, seed=args.seed
    )
    encoder = NewsEncoder(cfg, GloveTextEncoder(tokens, vocab_size=n, embedding_dim=d))
    user = UserEncoder(cfg, encoder)
    encoder.load_state_dict(
        {
            k[len("news_encoder.") :]: v
            for k, v in weights.items()
            if k.startswith("news_encoder.")
        },
        strict=True,
    )
    user.load_state_dict(
        {
            k[len("user_encoder.") :]: v
            for k, v in weights.items()
            if k.startswith("user_encoder.")
        },
        strict=True,
    )
    encoder.eval()
    user.eval()
    vectors = torch.zeros(len(tokens), d)
    for batch in torch.from_numpy(ids).split(64):
        vectors[batch] = encoder(batch, training=False)
    print(f"Encoded {len(ids)} unique articles", flush=True)
    user.news_encoder = CachedArticles(vectors)
    rows, history_indices = {b"": 0}, []
    with ZipFile(cohort) as z:
        for row in z.open("behaviors.tsv"):
            history = row.split(b"\t", 4)[3]
            history_indices.append(rows.setdefault(history, len(rows)))
    histories = torch.zeros(len(rows), 50, dtype=torch.long)
    for value, index in rows.items():
        valid = [int(t[1:]) for t in value.split()[-50:]]
        if valid:
            histories[index, -len(valid) :] = torch.tensor(valid)
    user_vectors = torch.cat(
        [user(batch, training=False) for batch in histories.split(64)]
    ).numpy()
    print(f"Encoded {len(rows)} unique histories", flush=True)
    vectors = vectors.numpy()
    flat = []
    with ZipFile(cohort) as z:
        for i, row in enumerate(z.open("behaviors.tsv")):
            items = [int(t[1:-2]) for t in row.rstrip(b"\n").split(b"\t", 4)[4].split()]
            flat.append(vectors[items] @ user_vectors[history_indices[i]])
    np.save(args.scores, np.concatenate(flat))


if __name__ == "__main__":
    main()
