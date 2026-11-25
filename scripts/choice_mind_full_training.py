"""Fit published implicit ALS and historical engagement controls on full MIND-large."""
import argparse
from array import array
from importlib import util
import json
from pathlib import Path
from zipfile import ZipFile

import implicit
from implicit.cpu.als import AlternatingLeastSquares
import numpy as np
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from choice_external_benchmark import METRIC_NAMES, ci, digest, grade, official_check
from choice_external_mind import evaluate
from choice_external_verify import EXPECTED
from choice_neural_gain import rank_scores
from choice_neural_gain_stream import read


def fit_inputs(archive, items):
    us, it, exposures = array("i"), array("i"), array("i")
    clicks = np.zeros(items, dtype=np.int64)
    shows = np.zeros_like(clicks)
    last_history, dates = {}, set()
    rows = 0
    with ZipFile(archive) as z:
        for line in z.open("behaviors.tsv"):
            _, user, time, history, slate = line.rstrip(b"\r\n").split(b"\t")
            user = int(user[1:])
            dates.add(time.split()[0].decode())
            if last_history.get(user) != history:
                values = [int(t[1:]) for t in history.split()]
                us.extend([user] * len(values))
                it.extend(values)
                last_history[user] = history
            for token in slate.split():
                item = int(token[1:-2])
                exposures.append(item)
                if token[-1:] == b"1":
                    clicks[item] += 1
                    us.append(user)
                    it.append(item)
            rows += 1
            if rows % 8192 == 0:
                shows += np.bincount(np.asarray(exposures), minlength=items)
                exposures = array("i")
    shows += np.bincount(np.asarray(exposures), minlength=items)
    matrix = csr_matrix((np.ones(len(it), dtype=np.float32), (np.asarray(us), np.asarray(it))),
                        shape=(max(last_history)+1, items))
    matrix.data.fill(1)
    return matrix, shows, clicks, dict(impressions=rows, candidate_exposures=int(shows.sum()),
             clicks=int(clicks.sum()), people=len(last_history), dates=sorted(dates),
             distinct_positive_user_items=matrix.nnz, observed_articles=int((shows>0).sum()))


def profiles(archive, width):
    histories = {}
    per_list, columns, pointers = array("i"), array("i"), array("i", [0])
    with ZipFile(archive) as z:
        for line in z.open("behaviors.tsv"):
            history = line.split(b"\t", 4)[3]
            if history not in histories:
                histories[history] = len(histories)
                columns.extend(sorted({int(t[1:]) for t in history.split()[-50:]}))
                pointers.append(len(columns))
            per_list.append(histories[history])
    features = csr_matrix((np.ones(len(columns), np.float32), np.asarray(columns), np.asarray(pointers)),
                          shape=(len(histories), width))
    return np.asarray(per_list), features


def als_scores(model, archive, raw, seen_in_training):
    index, history = profiles(archive, len(model.item_factors))
    factors = model.recalculate_user(np.arange(history.shape[0]), history)
    # The published model cannot estimate an embedding for an article with no positive evidence.
    model.item_factors[~seen_in_training] = 0
    scores = np.empty(len(raw.items), np.float32)
    for start in range(0, len(index), 1024):
        stop = min(start+1024, len(index))
        a, b = raw.offsets[start], raw.offsets[stop]
        first = factors[np.repeat(index[start:stop], np.diff(raw.offsets[start:stop+1]))]
        scores[a:b] = np.einsum("ij,ij->i", first, model.item_factors[raw.items[a:b]])
    return scores


def frozen_fusion(data, offsets, empirical_ctr):
    avg_naml = np.zeros(len(empirical_ctr), np.float32)
    for seed in (42, 123, 456):
        raw = np.load(data/f"gain_full_naml_{seed}_cached.npy", mmap_mode="r")
        avg_naml += rank_scores(raw, offsets)/3
    nrms = rank_scores(np.load(data/"gain_full_nrms.npy", mmap_mode="r"), offsets)
    local = rank_scores(-np.load(data/"ext_full_local50.npz")["ranks"], offsets)
    neural = (.35*avg_naml + .15*nrms + .20*local)/.70
    return .8*rank_scores(neural, offsets) + .2*rank_scores(empirical_ctr, offsets)


def run(args):
    p = json.loads(args.protocol.read_text())
    data = args.data
    train, dev = data/"MINDlarge_train.zip", data/"MINDlarge_dev.zip"
    if digest(train)!=p["train_sha256"] or digest(dev)!=p["validation_sha256"]:
        raise ValueError("Unexpected source archive")
    if implicit.__version__ != "0.7.2" or digest(data/"evaluate.py") != EXPECTED:
        raise ValueError("Unexpected implementation version")
    spec = util.spec_from_file_location("microsoft_evaluator", data/"evaluate.py")
    scorer = util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    width = 1
    for path in (train,dev):
        with ZipFile(path) as z:
            width = max(width, 1+max(int(row.split(b"\t",1)[0][1:]) for row in z.open("news.tsv")))
    matrix, shows, clicks, summary = fit_inputs(train, width)
    print("Read full training release",summary,flush=True)
    known = np.bincount(matrix.indices,minlength=width)>0
    model = AlternatingLeastSquares(factors=64, regularization=.1, alpha=40, iterations=15,
                                    random_state=575, num_threads=4)
    model.fit(matrix, show_progress=False)
    del matrix
    raw = read(dev)
    same = read(data/"gain_full_large.zip")
    if max(int(d.split("/")[1]) for d in summary["dates"]) >= int(raw.hours.min()//24):
        raise ValueError("Training and evaluation days overlap")
    for field in ("ids","users","hours","labels","offsets"):
        np.testing.assert_array_equal(getattr(raw,field),getattr(same,field))
    ctr = (clicks[raw.items] + 20*clicks.sum()/shows.sum())/(shows[raw.items]+20)
    scores = {"MostPop-large":clicks[raw.items], "SmoothedCTR-large":ctr,
              "ImplicitALS-large":als_scores(model,dev,raw,known),
              "ReelRank-large-prior":frozen_fusion(data,raw.offsets,ctr)}
    idx = np.sort(np.random.default_rng(575).choice(len(raw.ids),512,replace=False))
    tables, results = {}, {}
    for name, values in scores.items():
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite score")
        _, ranks = evaluate(values,same.ids,same.offsets,same.items,same.labels)
        tables[name] = grade(raw.labels,raw.offsets,ranks)
        np.save(data/f"ext_ranks_full_{name}.npy",ranks)
        tt = tables[name]
        results[name] = {**dict(zip(METRIC_NAMES,tt.mean(axis=0))),"first_hits":int(tt[:,4].sum()),
                         "official_random_audit":official_check(scorer,raw.ids,raw.offsets,raw.labels,ranks,idx,tt)}
        print(name,results[name],flush=True)
    for external in ("NAML-456","NAML-42","NRMS-456","LSTUR-123"):
        ranks = np.load(data/f"ext_ranks_full_{external}.npy", mmap_mode="r")
        tables[external] = grade(raw.labels,raw.offsets,ranks)
    _, groups = np.unique(raw.users,return_inverse=True)
    combined = tables["ReelRank-large-prior"]
    paired = {name:ci(combined-values,groups) for name,values in tables.items() if name!="ReelRank-large-prior"}
    report = dict(training=summary, training_sha256=p["train_sha256"],validation_sha256=p["validation_sha256"],
                  protocol_sha256=digest(args.protocol),implementation=implicit.__version__,
                  confirmation_impressions=len(raw.ids),confirmation_candidates=len(raw.labels),
                  validation_candidates_with_training_click_fraction=float(np.mean(known[raw.items])),
                  validation_candidates_with_logged_training_exposure_fraction=float(np.mean(shows[raw.items]>0)),
                  scores=results,paired_reader_intervals=paired)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("data","protocol","out"):
        parser.add_argument("--"+arg,required=True,type=Path)
    with threadpool_limits(limits=1,user_api="blas"):
        run(parser.parse_args())
