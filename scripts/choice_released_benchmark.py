"""Replay the released movie/book recommendation engines and ReelRank on original observed choices."""
import argparse
from pathlib import Path
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd
from scripts import choice_fusion as fusion

METHODS = ("identity","popularity","ease","implicit_mf","user_knn","std_vae","mlt_vae")
MODEL = "personal50"


def read_inputs(root):
    raw = pd.read_parquet(root/"data/behaviors.parquet")
    raw = raw[raw.iteration.notna()].set_index(["userId","impressionId"]).sort_index()
    methods = {}
    for method in METHODS:
        rank = pd.read_csv(root/f"recommendation_{method}_ranks.csv.gz").set_index(["userId","impressionId"]).ranking.sort_index()
        if not rank.index.is_unique or not rank.index.equals(raw.index):
            raise ValueError(f"Coverage mismatch for {method}")
        methods[method] = rank
    return raw, methods


def fixed_kaguya_id(root, movie_zip):
    with ZipFile(movie_zip) as z:
        movies=pd.read_csv(z.open("ml-25m/movies.csv"))
        links=pd.read_csv(z.open("ml-25m/links.csv"))
    chosen=movies[movies.title.str.contains("Kaguyahime no monogatari",regex=False)].merge(links)
    rows=pd.read_parquet(root/"data/movies.parquet")
    matches=rows[rows.imdbId.isin(chosen.imdbId)].movieId.unique()
    if len(matches)!=1: raise ValueError("Expected a unique catalog title for Kaguya")
    return int(matches[0])


def compare(raw, orders, model_path, film):
    model=pd.read_csv(model_path)
    model=model[model.method.eq(MODEL)].set_index(["user","slate"]).sort_index()
    if not model.index.is_unique: raise ValueError("Repeated evaluation list")
    all_orders={**orders,MODEL:model.ranking}
    discount=1/np.log2(np.arange(2,12))
    scored=[]; bins={}; examples=[]; empty=0
    for (uid,slate),r in model.iterrows():
        obs=raw.at[(uid,slate),"impressions"]
        shown=[int(e["itemId"]) for e in obs]
        selected={int(e["itemId"]) for e in obs if e["selected"]}
        if len(set(shown))!=10: raise ValueError("Expected ten unique candidates")
        empty += int(not selected)
        positions={}
        for method,rankings in all_orders.items():
            ranked=list(map(int,rankings.at[(uid,slate)].split(",")))
            if len(ranked)!=10 or set(ranked)!=set(shown) or (method=="identity" and ranked!=shown):
                raise ValueError(f"Ranked candidates differ from source: {method}")
            relevant=np.array([j in selected for j in ranked],dtype=bool)
            k=len(selected)
            gain=discount@relevant/discount[:k].sum() if k else 0.
            first=int(np.argmax(relevant)+1) if k else 11
            scored.append((uid,slate,r.domain,method,gain,relevant[:5].mean(),int(relevant[0]),int(relevant[:5].any()),first))
            source_pos=shown.index(ranked[0])+1
            b=0 if source_pos==1 else 1 if source_pos<=5 else 2
            hits=bins.setdefault(method,[0,0,0])
            hits[b]+=int(relevant[0])
            if selected=={film}: positions[method]=ranked.index(film)+1
        if r.domain=="movies" and positions.get("identity")==6 and positions.get(MODEL)==1:
            examples.append(positions)
    cols=["user","slate","domain","method","ndcg","p5","hit1","hit5","first"]
    frame=pd.DataFrame(scored,columns=cols)
    metrics=frame.groupby("method")[cols[4:]].mean().to_dict(orient="index")
    counts={k:int(v.hit1.sum()) for k,v in frame.groupby("method")}
    pivot=frame.pivot(index=["user","slate"],columns="method",values="hit1")
    changes={m:{"gained":int((pivot[MODEL]>pivot[m]).sum()),"lost":int((pivot[MODEL]<pivot[m]).sum())} for m in METHODS}
    thresholds={m:{str(k):int((g["first"]<=k).sum()) for k in [1,2,3,5]} for m,g in frame.groupby("method")}
    domain={k:v.groupby("method")[["ndcg","p5","hit1"]].mean().to_dict(orient="index") for k,v in frame.groupby("domain")}
    diff=frame.pivot(index=["user","slate"],columns="method",values="hit1")
    by_user=(diff[MODEL]-diff.user_knn).groupby(level="user").mean()
    return {"lists":len(model),"people":len(by_user),"empty":empty,
            "metrics":metrics,"hits":counts,"paired":fusion.intervals(frame,MODEL),"transitions":changes,
            "by_original_position":bins,"positions_order":["1","2-5","6-10"],
            "cumulative":thresholds,"domains":domain,
            "participants_better_tied_worse_vs_userknn":[int((by_user>0).sum()),int((by_user==0).sum()),int((by_user<0).sum())],
            "kaguya_matching_lists":examples}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("root","private","movie-zip","out"):
        parser.add_argument("--"+arg,type=Path,required=True)
    args=parser.parse_args()
    raw,methods=read_inputs(args.root)
    film=fixed_kaguya_id(args.root,args.movie_zip)
    report={"released_orders":list(METHODS),"full_genuine_ten_title_lists":len(raw)}
    for stage,name in (("confirmation","confirmation"),("grouped","grouped_cv"),("forward","calendar")):
        report[stage]=compare(raw,methods,args.private/f"personal_{name}.csv.gz",film)
        print(stage,report[stage]["hits"],report[stage]["kaguya_matching_lists"],flush=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
