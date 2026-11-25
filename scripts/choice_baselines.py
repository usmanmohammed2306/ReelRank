"""Reproduce the SM-RS 2.0 Task 3 baselines from publisher checkpoints.

Legacy uses cell 12 of the OSF notebook literally. Recommendation scores only
iteration-tagged ten-item slates using strictly earlier, timestamped selections.
"""
import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from scripts.fetch_choice_osf import MANIFEST, digest

CODE_COMMIT = "3dcc7d36facf026798696d7b891663ca40ecb511"
DATA_COMMIT = "3c8af3b828d9a194d15f1b19abb739319cdb1751"
PAPER = {"identity": [.215,.131], "popularity": [.406,.244], "ease": [.368,.225],
         "implicit_mf": [.337,.210], "user_knn": [.367,.224],
         "std_vae": [.292,.174], "mlt_vae": [.328,.189]}
N_ITEMS = {"movies":15876, "books":6548}


@dataclass
class Case:
    user: int
    slate: int
    domain: str
    ids: np.ndarray
    true: list
    past: np.ndarray
    shuffled: np.ndarray


def cases(root, mode):
    behavior = pd.read_parquet(root / "data/behaviors.parquet")
    domains = pd.read_parquet(root / "data/users.parquet").set_index("userId").domain
    rng = np.random.RandomState(2024)
    out = []
    for user, group in behavior.groupby("userId", sort=False):
        domain = domains.at[user]
        history = []
        first = group.iloc[0]
        if mode == "legacy":
            for i,row in enumerate(group.itertuples()):
                if i:
                    ids = np.array([int(x["itemId"]) for x in first.impressions])
                    chosen = [int(x["itemId"]) for x in first.impressions if x["selected"]]
                    shuffled = ids.copy()
                    rng.shuffle(shuffled)
                    out.append(Case(user, int(row.impressionId), domain, ids, chosen,
                                    np.asarray(history, dtype=int), shuffled))
                history.extend(int(x["itemId"]) for x in row.impressions if x["selected"])
        else:
            events = []
            for record in group.itertuples():
                at = record.timestamp.value / 1e9
                for item in record.impressions:
                    clicked = item["timestamp"]
                    if item["selected"] and pd.notna(clicked):
                        events.append((max(at, float(clicked)), int(item["itemId"])))
            events.sort()
            event_times = np.array([e[0] for e in events])
            for row in group.itertuples():
                if pd.isna(row.iteration):
                    continue
                ids = np.array([int(x["itemId"]) for x in row.impressions])
                if len(ids) != 10 or len(set(ids)) != 10:
                    raise ValueError("Expected ten distinct alternatives")
                chosen = [int(x["itemId"]) for x in row.impressions if x["selected"]]
                n_past = np.searchsorted(event_times, row.timestamp.value / 1e9, side="left")
                past = np.unique([v[1] for v in events[:n_past]]).astype(int)
                shuffled = ids.copy()
                rng.shuffle(shuffled)
                out.append(Case(user,int(row.impressionId),domain,ids,chosen,past,shuffled))
    return out


def _load(path):
    if digest(path) != MANIFEST[path.name]["sha256"]:
        raise ValueError(f"OSF checksum mismatch: {path.name}")
    return np.load(path,mmap_mode="r",allow_pickle=False)


def rank_baseline(name, rows, artifacts, mode):
    if name == "identity":
        return [r.ids for r in rows]
    out = [None] * len(rows)
    for domain, n_items in N_ITEMS.items():
        indices = [i for i,r in enumerate(rows) if r.domain == domain]
        offset = 0 if domain == "movies" else N_ITEMS["movies"]
        if name == "popularity":
            weights = _load(artifacts/("mean_popularities_"+domain+".npy"))
            # This uses the exact tie order of argsort over the full catalog,
            # as in notebook cell 15, without scanning the whole catalog each time.
            pos = np.argsort(np.argsort(-weights))
            for i in indices:
                r=rows[i]
                ids=np.unique(r.ids)
                out[i]=ids[np.argsort(pos[ids-offset])]
        elif name == "ease":
            B = _load(artifacts/("full_"+domain+"_item_item.npy"))
            for i in indices:
                r=rows[i]
                ids=np.unique(r.ids)
                past = np.unique(r.past) - offset
                # The matrix-vector product restricted to candidate columns.
                if len(past):
                    vals=B[np.ix_(past,ids-offset)].sum(axis=0)
                    if len(set(vals)) == len(vals):
                        out[i] = ids[np.argsort(-vals)]
                        continue
                # Original argsort tie order depends on the full catalog.
                vals = B[past].sum(axis=0) if len(past) else np.zeros(n_items)
                order=np.argsort(-vals)
                out[i] = order[np.isin(order,ids-offset)] + offset
        elif name in ["implicit_mf","user_knn"]:
            import binpickle
            path=artifacts/(name+"_"+domain+".bpk")
            if digest(path)!=MANIFEST[path.name]["sha256"]:
                raise ValueError("Unexpected LensKit checkpoint digest")
            model=binpickle.load(path)
            for i in indices:
                r=rows[i]
                raw=r.past-offset
                # Published notebook calls pd.Series(history), so LensKit interprets
                # sequence positions as item ids. Corrected branch supplies real ids.
                if mode == "legacy":
                    profile=pd.Series(raw)
                else:
                    profile=pd.Series(1., index=np.unique(raw),dtype=float)
                if name == "implicit_mf":
                    # Older LensKit 0.14.2 allocates a mis-sized array when the
                    # query includes unsupported ids. Filter explicitly; no
                    # trained item feature exists for these observations.
                    profile=profile[profile.index.isin(model.predictor.item_index_)]
                predictions=model.predict_for_user(None,r.shuffled-offset,profile)
                out[i]=np.asarray(predictions.sort_values(ascending=False).index)+offset
        else:
            out_domain = vae_predictions(name, domain, [rows[i] for i in indices], artifacts, offset)
            for i,rank in zip(indices,out_domain):
                out[i] = rank
    return out


def vae_predictions(name, domain, rows, artifacts, offset):
    from recommenders.models.vae.standard_vae import StandardVAE
    from recommenders.models.vae.multinomial_vae import Mult_VAE
    import tensorflow as tf
    filename = name+"_"+domain+"_cache.zip"
    archive=artifacts/filename
    if digest(archive)!=MANIFEST[filename]["sha256"]:
        raise ValueError("Unexpected VAE checkpoint digest")
    directory=filename[:-4]
    with ZipFile(archive) as z:
        for member in ["tf_model_weights.index","tf_model_weights.data-00000-of-00001"]:
            path=directory+"/"+member
            z.extract(path,artifacts)
    n=N_ITEMS[domain]
    common=dict(n_users=27410 if domain=="movies" else 53424,original_dim=n,
                intermediate_dim=200,latent_dim=100,n_epochs=50,batch_size=100,
                k=100,verbose=0,save_path=None,drop_encoder=.5,drop_decoder=.5)
    tf.keras.utils.set_random_seed(2024)
    if name=="std_vae":
        model=StandardVAE(**common,annealing=False,beta=1.0)
    else:
        model=Mult_VAE(**common,seed=2024,annealing=False,beta=(.3 if domain=="movies" else .35))
    status=model.model.load_weights(str(artifacts/directory/"tf_model_weights"))
    status.assert_existing_objects_matched().expect_partial()
    out=[]
    for start in range(0,len(rows),32):
        batch=rows[start:start+32]
        x=np.zeros((len(batch),n),dtype=np.float32)
        for j,r in enumerate(batch):
            x[j,r.past-offset]=1
        # Same Keras inference graph as the notebook, in batches. The source
        # layer samples a latent vector even at inference, so results can vary
        # from the paper by seed, batch shape and TensorFlow version.
        pred=model.model(x,training=False).numpy()
        for vals,r in zip(pred,batch):
            ids=np.unique(r.ids)
            order=np.argsort(-vals)
            out.append(order[np.isin(order,ids-offset)]+offset)
    return out


def summarize(rows,ranks,mode,per_record):
    from smrs import metrics
    from smrs.tasks.task3_reranking import evaluate
    targets=[r.true for r in rows]
    ndcg=[metrics.ndcg_at_k(rank,s) for rank,s in zip(ranks,targets)]
    precision=[metrics.precision_at_k(rank,s) for rank,s in zip(ranks,targets)]
    if mode=="legacy":
        # The archive uses min(5, slate length) and ideal DCG over scored rows.
        def archive_scores(rank,t):
            rel=np.isin(rank,t)
            k=min(5,len(rank))
            disc=1/np.log2(np.arange(2,min(len(rank),10)+2))
            dcg=np.sum(rel[:10]*disc)
            idcg=disc[:min(10,int(rel.sum()))].sum()
            return [dcg/idcg if idcg else 0., rel[:k].mean()]
        legacy=np.asarray([archive_scores(r,s) for r,s in zip(ranks,targets)])
    out={"n_slates":len(rows),"n_users":len({r.user for r in rows}),
         "canonical":evaluate(ranks,targets)}
    if mode=="legacy":
        means=legacy.mean(axis=0)
        out["notebook_metrics"]={"nDCG@10":means[0],"Precision@5":means[1]}
    if per_record:
        df=pd.DataFrame({"impressionId":[r.slate for r in rows],"userId":[r.user for r in rows],
                         "domain":[r.domain for r in rows],"ndcg":ndcg,"p5":precision})
        df.to_pickle(per_record)
        rank_file=per_record.with_name(per_record.name.replace("_rows.pkl","_ranks.csv.gz"))
        pd.DataFrame({"userId":df.userId,"impressionId":df.impressionId,
                      "ranking":[",".join(map(str,rank)) for rank in ranks]}).to_csv(rank_file,index=False)
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("--methods",nargs="+",choices=PAPER,default=list(PAPER))
    p.add_argument("--modes",nargs="+",choices=["legacy","recommendation"],default=["legacy","recommendation"])
    p.add_argument("--out",type=Path,default=Path("results/choice"))
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    for mode in args.modes:
        rows=cases(args.source,mode)
        for name in args.methods:
            ranks=rank_baseline(name,rows,args.source/"artifacts",mode)
            details=(args.source/(mode+"_"+name+"_rows.pkl")) if mode=="recommendation" else None
            res=summarize(rows,ranks,mode,details)
            res.update(mode=mode,model=name,published_mixed_table=dict(zip(["nDCG@10","Precision@5"],PAPER[name])),
                       evaluator_commit=CODE_COMMIT,dataset_commit=DATA_COMMIT)
            f=args.out/(mode+"_"+name+".json")
            f.write_text(json.dumps(res,indent=2,allow_nan=False)+"\n")
            print(mode,name,res,flush=True)
