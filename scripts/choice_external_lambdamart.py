"""Train the published LambdaMART algorithm with the same features and supervision as ReelRank."""
import argparse
from itertools import product
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from scripts import choice_content_ranker as base
from scripts import choice_fusion as fusion
from scripts import choice_personal_ranker as personal


def matrix(cases, bank):
    return np.concatenate((base.concatenate(cases, bank), np.stack([c.coords for c in cases]),
                           np.stack([c.personal for c in cases])), axis=2)


def prepare(train, test):
    if {c.user for c in train} & {c.user for c in test}:
        raise ValueError("Overlap between training and validation people")
    bank = base.Uptake(train)
    x = matrix(train, bank)
    y = np.stack([c.y for c in train]).astype(int)
    order = np.stack([np.argsort(fusion.tie_key(c.ids, c.slate)) for c in train])
    row = np.arange(len(train))[:,None]
    return x[row,order].reshape(-1,x.shape[-1]), y[row,order].ravel(), matrix(test,bank)


def predict(arrays, cfg):
    leaves, rounds, cutoff = cfg
    x, y, q = arrays
    estimator = lgb.LGBMRanker(objective="lambdarank", label_gain=[0,1],
                               lambdarank_truncation_level=cutoff, num_leaves=leaves,
                               n_estimators=rounds, learning_rate=.05, reg_lambda=5,
                               min_child_samples=30, max_bin=63, random_state=575,
                               n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True)
    estimator.fit(x,y,group=np.full(len(y)//10,10))
    return estimator.predict(q.reshape(-1,q.shape[-1])).reshape(len(q),10)


def evaluate(train, test, reference, cfg, private, stage):
    scores = predict(prepare(train,test), cfg)
    rows = base.rows_for(test, {"lambdamart":scores})
    confirmed = reference[reference.method=="personal50"].copy()
    if set(confirmed.slate)!=set(rows.slate):
        raise ValueError("Models evaluated on different slates")
    joined = pd.concat([rows,confirmed],ignore_index=True)
    base.canonical(test,joined)
    rows.to_csv(private/f"lambdamart_{stage}.csv.gz", index=False)
    report = base.report(joined)
    report["paired"] = fusion.intervals(joined,"personal50")["lambdamart"]
    report["first_hits"] = {k:int(v.hit1.sum()) for k,v in joined.groupby("method")}
    return report


def run(args):
    if lgb.__version__ != "4.6.0":
        raise RuntimeError("Use the registered LightGBM version")
    if not args.out.joinpath("lambdamart_protocol.json").is_file():
        raise ValueError("Missing comparator protocol")
    users=pd.read_parquet(args.root/"data/users.parquet").userId.tolist()
    print("Building identical feature tensors",flush=True)
    cases=personal.assemble(args.root,args.movie_zip,users)
    dev=[c for c in cases if fusion.participant_fold(c.user)==0]
    test=[c for c in cases if fusion.participant_fold(c.user)!=0]
    grid=list(product((7,15,31),(80,200),(3,10)))
    all_scores={cfg:[] for cfg in grid}
    for fold,(tr,va) in enumerate(GroupKFold(4).split(dev,groups=[c.user for c in dev])):
        tc=[dev[i] for i in tr];vc=[dev[i] for i in va]
        arrays=prepare(tc,vc)
        for cfg in grid:
            scored=base.rows_for(vc,{"lambdamart":predict(arrays,cfg)})
            all_scores[cfg].append(scored)
        print("Finished development fold",fold+1,flush=True)
    summary={cfg:pd.concat(rows)[["hit1","ndcg"]].mean().to_dict() for cfg,rows in all_scores.items()}
    best=max(grid,key=lambda c:(summary[c]["hit1"],summary[c]["ndcg"],-c[0],-c[1],-c[2]))
    result={"version":lgb.__version__,"features":int(arrays[0].shape[1]),
            "development_people":len({c.user for c in dev}),"development_slates":len(dev),
            "settings":list(best),"grid":{"-".join(map(str,k)):v for k,v in summary.items()},
            "protocol_sha256":base.digest(args.out/"lambdamart_protocol.json")}
    (args.out/"lambdamart_lock.json").write_text(json.dumps(result,indent=2)+"\n")
    reference=pd.read_csv(args.private/"personal_confirmation.csv.gz")
    result["confirmation"]=evaluate(dev,test,reference,best,args.private,"confirmation")
    print(result["confirmation"]["first_hits"],flush=True)
    from scripts.choice_personal_forward import time_split
    logs = pd.read_parquet(args.root/"data/behaviors.parquet")
    _, earlier, later, _ = time_split(logs)
    future = pd.read_csv(args.private/"personal_calendar.csv.gz")
    result["forward"] = evaluate([c for c in cases if c.user in earlier],
                                 [c for c in cases if c.user in later],future,best,args.private,"forward")
    print("Completed primary comparison",flush=True)
    (args.out/"lambdamart_results.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("root","private","out","movie-zip"):
        p.add_argument("--"+flag,type=Path,required=True)
    with threadpool_limits(limits=4):
        run(p.parse_args())
