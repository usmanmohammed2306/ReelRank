"""Prepare ranked inputs for the frozen and delayed-feedback news models."""
from pathlib import Path

import numpy as np
from numba import njit

from choice_external_mind import to_percentile

ROOT=Path(__file__).resolve().parents[1]
@njit
def rank_scores(scores,offsets):
    out=np.empty(len(scores),dtype=np.float32)
    for a,b in zip(offsets[:-1],offsets[1:]):
        out[a:b]=to_percentile(scores[a:b])
    return out


def fields(data,part,p):
    def load(tag):
        return np.load(data/f"gain_{part}_{tag}.npy",mmap_mode="r")
    nrms_file=data/("nrms_scores.npy" if part=="small" else "gain_full_nrms.npy")
    nrms_raw=np.load(nrms_file,mmap_mode="r")
    local_file=data/("full_ranks.npz" if part=="small" else "gain_full_local.npz")
    local_order=np.load(local_file)["ranks"]
    raw_seeds=[load(f"naml_{s}_cached") for s in (42,123,456)]
    for s in [nrms_raw,local_order,*raw_seeds]:
        if len(s)!=len(p.items) or not np.isfinite(s).all():
            raise ValueError("Missing or nonfinite rankings")
    rank=rank_scores(nrms_raw,p.offsets)
    local=rank_scores(-local_order,p.offsets)
    naml=np.zeros(len(rank),np.float32)
    for s in raw_seeds: naml+=rank_scores(s,p.offsets)/3
    variants={n:rank_scores(load(n),p.offsets) for n in ("live","frozen")}
    return dict(nrms=rank,nrms_raw=nrms_raw,local_order=local_order,local=local,
                naml=naml,seeds=raw_seeds,**variants)


def fixed(f,telemetry):
    return .35*f["naml"]+.15*f["nrms"]+.2*f["local"]+.3*f[telemetry]
