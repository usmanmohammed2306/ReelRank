"""Regrade released rankings, paper numbers and a separately trained supervised comparator."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.choice_released_benchmark import METHODS


def verify(args):
    def load(path):
        return json.loads(path.read_text())
    original = pd.read_parquet(args.root/"data/behaviors.parquet")
    original = original[original.iteration.notna()].set_index(["userId","impressionId"])
    lm = load(args.results/"lambdamart_results.json")
    external = load(args.results/"external_baseline_audit.json")
    disc = 1/np.log2(np.arange(2,12))
    grades = {}
    for key, suffix in (("confirmation","confirmation"),("forward","calendar")):
        refs = pd.read_csv(args.private/f"personal_{suffix}.csv.gz").query("method == 'personal50'")
        other = pd.read_csv(args.private/f"lambdamart_{key}.csv.gz")
        if set(zip(refs.user,refs.slate)) != set(zip(other.user,other.slate)):
            raise ValueError("Differing held-out lists")
        groups = pd.concat([refs,other]).groupby("method")
        obs = {}
        for method,group in groups:
            if group.duplicated(["user","slate"]).any():
                raise ValueError("Duplicate list")
            values = []
            for r in group.itertuples():
                history = original.at[(r.user,r.slate),"impressions"]
                choices = {int(o["itemId"]):bool(o["selected"]) for o in history}
                ranked = list(map(int,r.ranking.split(",")))
                if len(ranked)!=10 or set(choices)!=set(ranked):
                    raise ValueError("Prediction candidates do not match raw display")
                y = np.array([choices[i] for i in ranked])
                where = np.flatnonzero(y)
                dcg = disc[where].sum()/disc[:len(where)].sum() if len(where) else 0
                val = [dcg,y[:5].mean(),float(y[0]),float(y[:5].any()),where[0]+1 if len(where) else 11]
                np.testing.assert_allclose(val,[r.ndcg,r.p5,r.hit1,r.hit5,r.first],rtol=0,atol=1e-12)
                values.append(val)
            mean = np.mean(values,axis=0)
            names = ("ndcg","p5","hit1","hit5","first")
            expected = [lm[key]["metrics"][method][x] for x in names]
            np.testing.assert_allclose(expected,mean,rtol=0,atol=1e-12)
            obs[method]=dict(first_hits=int(np.sum(np.array(values)[:,2])), lists=len(values),
                             raw_metrics=dict(zip(names,mean)))
        grades[key]=obs
    record = {"raw_label_regrade":grades,"original_release_rerun":{}}
    counts = [0,0]
    for name in METHODS:
        filename = "recommendation_"+name+"_ranks.csv.gz"
        old = pd.read_csv(args.root/filename)
        new = pd.read_csv(args.rerun/filename)
        ids = ["userId","impressionId"]
        old, new = (f.set_index(ids).sort_index().ranking for f in (old,new))
        if not old.index.equals(new.index) or len(new)!=len(original):
            raise ValueError("External baseline coverage changed")
        if not new.index.equals(original.index.sort_values()):
            raise ValueError("Source impression identity mismatch")
        paper_path = "legacy_"+name+".json"
        fresh = load(args.rerun/"results"/paper_path)
        first = load(args.original_aggregates/paper_path)
        target = fresh["published_mixed_table"]
        matches = [sum(round(m["notebook_metrics"][k],3)==t for k,t in target.items()) for m in (first,fresh)]
        counts = [a+b for a,b in zip(counts,matches)]
        record["original_release_rerun"][name] = dict(
            original_notebook_prefixes=fresh["n_slates"],observed_recommendation_lists=len(new),
            identical_rank_orders=int((new==old).sum()),original_paper=target,
            first_replay=first["notebook_metrics"],fresh_replay=fresh["notebook_metrics"],
            paper_cells_matching_first_and_fresh=matches)
    record["printed_cells_reproduced"]={"total":14,"first_replay":counts[0],"fresh_replay":counts[1]}
    if args.refit:
        check = {}
        for key,suffix in (("confirmation","confirmation"),("grouped","grouped_cv"),("forward","calendar")):
            old = pd.read_csv(args.private/f"personal_{suffix}.csv.gz").query("method=='personal50'")
            new = pd.read_csv(args.refit/f"personal_{suffix}.csv.gz")
            left,right=(f.set_index(["user","slate"]).sort_index() for f in (old,new))
            if not left.index.equals(right.index):
                raise ValueError("Refit produced different cohort")
            check[key]={"lists":len(right),"identical_top_ten_orders":int((left.ranking==right.ranking).sum()),
                        "first_hits":int(right.hit1.sum()),"archived_first_hits":external[key]["hits"]["personal50"]}
        record["isolated_frozen_refit"]=check
    if args.repeat_ltr_results:
        if not args.refit:
            raise ValueError("Repeated LambdaMART needs its private refit rankings")
        lock0=load(args.results/"lambdamart_lock.json")
        lock1=load(args.repeat_ltr_results/"lambdamart_lock.json")
        if lock0["settings"]!=lock1["settings"] or set(lock0["grid"])!=set(lock1["grid"]):
            raise ValueError("Development configuration changed in the independent run")
        delta=max(abs(v-lock1["grid"][key][metric]) for key,row in lock0["grid"].items() for metric,v in row.items())
        repeated={"max_grid_difference":delta,"winning_configuration":lock1["settings"]}
        for key in ("confirmation","forward"):
            a,b=(pd.read_csv(path/f"lambdamart_{key}.csv.gz").set_index(["user","slate"]).sort_index()
                 for path in (args.private,args.refit))
            if not a.index.equals(b.index) or not (a.ranking==b.ranking).all():
                raise ValueError("Independent LightGBM refit changed predicted orders")
            repeated[key]={"identical_rankings":len(b),"first_hits":int(b.hit1.sum())}
        record["isolated_lambdamart_repeat"]=repeated
    args.out.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("root","private","rerun","results","original-aggregates","out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--refit",type=Path)
    p.add_argument("--repeat-ltr-results",type=Path)
    verify(p.parse_args())
