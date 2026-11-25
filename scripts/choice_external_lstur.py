"""Run a pinned NewsReX LSTUR checkpoint on released, labelled MIND lists."""
import argparse
import hashlib
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from safetensors.torch import load_file

from choice_external_nrms import CachedArticles
from choice_neural_gain_naml import packed_data


@torch.inference_mode()
def score(path, destination, encoder, vectors, anonymous):
    # Histories can recur on multiple lists. The long-term ID is part of the cache key.
    keys = {}
    row_keys = []
    with ZipFile(path) as archive:
        for row in archive.open("behaviors.tsv"):
            _, uid, _, history, _ = row.split(b"\t", 4)
            key = (0 if anonymous else int(uid[1:]), history)
            row_keys.append(keys.setdefault(key, len(keys)))
    users = np.empty((len(keys), encoder.config.gru_unit), dtype=np.float32)
    keys = list(keys)
    for start in range(0, len(keys), 96):
        group = keys[start:start + 96]
        histories = torch.zeros(len(group), 50, dtype=torch.long)
        for i, (_, history) in enumerate(group):
            recent = [int(token[1:]) for token in history.split()[-50:]]
            if recent:
                histories[i, -len(recent):] = torch.tensor(recent)
        ids = torch.tensor([uid for uid, _ in group])
        users[start:start + len(group)] = encoder([histories, ids], training=False).numpy()
    scores = []
    news = vectors.numpy()
    with ZipFile(path) as archive:
        for row_id, row in enumerate(archive.open("behaviors.tsv")):
            _, _, _, _, candidates = row.rstrip().split(b"\t")
            item_ids = [int(token[1:-2]) for token in candidates.split()]
            scores.append(news[item_ids] @ users[row_keys[row_id]])
    np.save(destination, np.concatenate(scores))
    print("Scored", path.name, len(row_keys), "impressions", flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--seed", type=int, choices=(42, 123, 456), default=123)
    args = parser.parse_args()
    data = args.data
    path = data / f"LSTUR_{args.seed}_model.safetensors"
    expected = {
        42: "50768b2009fec9ff3efe34997ba1537b1f5f51cc9ebd58701b2378f424b673a9",
        123: "133141b4e6d224793b1128448cb9d84d306dc982f8dc1606037b535463a73714",
        456: "3c834a94219841091a54041b08d46b9f3729fb73037379aff8415e297bb07270"}[args.seed]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Incorrect LSTUR seed {args.seed} weights")
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.upstream.resolve()))
    from src.core.models.configs import LSTURConfig
    from src.frameworks.pytorch.layers import GloveTextEncoder
    from src.frameworks.pytorch.models.lstur import NewsEncoder, UserEncoder

    full = data / "gain_full_large.zip"
    title, _, packed, ids, vocabulary, cats, subcats = packed_data(data, full)
    cfg = LSTURConfig(cnn_filter_num=400, gru_unit=600, max_title_length=32,
                      use_category=True, use_subcategory=True, seed=args.seed)
    weights = load_file(str(path))
    news = NewsEncoder(cfg, GloveTextEncoder(title, vocab_size=vocabulary, embedding_dim=300),
                       cats, subcats)
    user = UserEncoder(cfg, news, len(weights["user_encoder.user_embedding.weight"]))
    for model, prefix in [(news, "news_encoder."), (user, "user_encoder.")]:
        model.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}, strict=True)
        model.eval()
    vectors = torch.zeros(len(title), news.output_dim)
    for batch in torch.cat([torch.tensor([0]), ids]).split(96):
        vectors[batch] = news(packed[batch], training=False)
    print("Encoded", len(ids), "articles", flush=True)
    user.news_encoder = CachedArticles(vectors)
    if torch.count_nonzero(user.user_embedding.weight[0]):
        raise ValueError("Expected index 0 to be an untrained zero vector")
    score(data / "MINDsmall_dev.zip", data / f"gain_small_lstur_{args.seed}.npy", user, vectors, False)
    # Small/large pseudonyms do not give an established account-ID mapping.
    score(full, data / f"gain_full_lstur_{args.seed}.npy", user, vectors, True)


if __name__ == "__main__":
    main()
