"""Separate first-position choice from ranking the remainder of a news list."""
import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from choice_external_mind import evaluate
from choice_neural_gain import fields, fixed
from choice_neural_gain_stream import read


@njit
def differences(components,static,live,rs,rl,labels,offsets):
    n=len(offsets)-1
    features=np.zeros((n,23),dtype=np.float32)
    outcomes=np.zeros(n,dtype=np.int8)
    top=np.zeros((n,2),dtype=np.int32)
    for i in range(n):
        a,b=offsets[i:i+2]
        left=a+np.argmin(rs[a:b]); right=a+np.argmin(rl[a:b])
        top[i]=np.array([left,right])
        outcomes[i]=int(labels[right])-int(labels[left])
        for j in range(5):
            x=components[left,j]; y=components[right,j]
            features[i,j]=x; features[i,j+5]=y; features[i,j+10]=y-x; features[i,j+15]=(y-x)**2
        features[i,20]=static[left]-static[right]
        features[i,21]=live[right]-live[left]
        features[i,22]=np.log(b-a)
    return features,outcomes,top


def fit(x,y,hours):
    out={}
    for h in range(0,24,4):
        valid=(hours<=360+h-2)&(y!=0)
        if np.count_nonzero(valid)<200: continue
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=200))
        model.fit(x[valid],y[valid]>0)
        scale,reg=model.steps[0][1],model.steps[1][1]
        out[h]={"n_donor_discordant":int(valid.sum()),"latest":int(hours[valid].max()),"mu":scale.mean_.tolist(),
                "sd":scale.scale_.tolist(),"beta":reg.coef_[0].tolist(),"intercept":float(reg.intercept_[0])}
    return out


def run(args):
    d=args.data;small=args.part=="small"
    p=read(d/("MINDsmall_dev.zip" if small else "gain_full_large.zip"))
    f=fields(d,args.part,p)
    static=(.35*f["naml"]+.15*f["nrms"]+.2*f["local"])/.7
    live=fixed(f,"live")
    _,rs=evaluate(static,p.ids,p.offsets,p.items,p.labels)
    _,rl=evaluate(live,p.ids,p.offsets,p.items,p.labels)
    comps=np.column_stack([f[k] for k in ("naml","nrms","local","live","frozen")])
    x,y,top=differences(comps,static,live,rs,rl,p.labels,p.offsets)
    del f,comps
    fitpath=args.out/"neural_first_coefficients.json"
    if small and not args.use_fitted:
        fitted=fit(x,y,p.hours)
        fitpath.write_text(json.dumps(fitted,indent=2)+"\n")
    else: fitted=json.loads(fitpath.read_text())
    use=np.zeros(len(y),dtype=bool)
    for h,vals in fitted.items():
        h=int(h); mask=(p.hours>=360+h)&(p.hours<360+h+4)
        if not mask.any():continue
        z=(x[mask]-vals["mu"])/vals["sd"]
        logit=z @ vals["beta"]+vals["intercept"]
        use[mask]=logit>np.log(.55/.45)
    sel=top[:,0].copy(); sel[use]=top[:,1][use]
    scores = live.copy()
    scores[sel] = 2.
    _, ranks = evaluate(scores, p.ids, p.offsets, p.items, p.labels)
    np.savez_compressed(d/f"gain_{args.part}_Switch_ranks.npz", ranks=ranks)
    print(args.part, "Delayed-feedback rankings", len(p.ids), flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--part",choices=["small","full"],required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--use-fitted",action="store_true",help="Use the supplied coefficient file for both splits.")
    run(p.parse_args())
