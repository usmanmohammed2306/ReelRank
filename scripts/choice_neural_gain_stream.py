"""Build delayed streaming article-posterior features exclusively from MIND-small logs."""
import argparse
from array import array
from collections import namedtuple
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from numba import njit

Period=namedtuple("Period","ids users hours offsets items labels")


@lru_cache(600000)
def hour_key(value):
    date, clock, ampm=value.split()
    month,day,year=map(int,date.split(b"/"))
    if year!=2019 or month!=11:
        raise ValueError("Expected the November 2019 MIND releases")
    h=int(clock.split(b":",1)[0])%12 + int(ampm==b"PM")*12
    return day*24+h


def read(path):
    ids,users,hours,offsets,items,ys=(array("i"),array("i"),array("i"),array("i",[0]),array("i"),bytearray())
    with ZipFile(path) as z:
        for line in z.open("behaviors.tsv"):
            sid,uid,time,_,cand=line.rstrip(b"\r\n").split(b"\t")
            ids.append(int(sid));users.append(int(uid[1:]));hours.append(hour_key(time))
            q=cand.split()
            items.extend(int(s[1:-2]) for s in q)
            ys.extend(s[-1]-48 for s in q)
            offsets.append(len(items))
    return Period(*(np.frombuffer(v,dtype=np.int32) for v in (ids,users,hours,offsets,items)),np.frombuffer(ys,dtype=np.uint8))


def article_categories(paths):
    values={};cats={}
    for path in paths:
        with ZipFile(path) as z:
            for line in z.open("news.tsv"):
                raw,category,*_=line.split(b"\t",2)
                i=int(raw[1:]); idx=cats.setdefault(category,len(cats))
                if values.setdefault(i,idx)!=idx:
                    raise ValueError("Conflicting canonical article category")
    ret=np.zeros(max(values)+1,dtype=np.int32)
    for i,k in values.items(): ret[i]=k
    return ret


@njit
def increments(hours,offsets,items,labels,zero,exposures,clicks):
    for j,hour in enumerate(hours):
        h=hour-zero
        for ix in range(offsets[j],offsets[j+1]):
            article=items[ix]
            exposures[h,article]+=1
            clicks[h,article]+=labels[ix]


def posteriors(exposures,clicks,categories,zero,query_hours,lag):
    """Only counts from source-hour bins ending at least lag hours earlier."""
    last=-1; nitems=len(categories)
    states=[(np.zeros(nitems),np.zeros(nitems)) for _ in (3,24)]
    out={}
    for t in sorted(set(query_hours)):
        cutoff=t-zero-lag-1
        while last<cutoff:
            last+=1
            for life,(n,k) in zip((3,24),states):
                decay=0.5**(1/life);n*=decay;k*=decay
                if 0<=last<len(exposures):
                    n+=exposures[last];k+=clicks[last]
        scores=np.zeros(nitems)
        for n,k in states:
            pop=(k.sum()+1)/(n.sum()+20)
            counts=np.bincount(categories,weights=n)
            hits=np.bincount(categories,weights=k)
            base=(hits+100*pop)/(counts+100)
            scores+=.5*(k+30*base[categories])/(n+30)
        out[int(t)]=scores.astype(np.float32)
    return out


def prepare(data):
    train=read(data/"MINDsmall_train.zip"); donors=read(data/"MINDsmall_dev.zip")
    cat=article_categories([data/"MINDsmall_train.zip",data/"MINDsmall_dev.zip",data/"gain_full_large.zip"])
    zero=int(train.hours.min());end=int(donors.hours.max())
    state=(end-zero+1,len(cat))
    n=np.zeros(state,dtype=np.int32);k=np.zeros(state,dtype=np.int32)
    increments(train.hours,train.offsets,train.items,train.labels,zero,n,k)
    hours=range(15*24,16*24)
    frozen=posteriors(n,k,cat,zero,hours,1)
    increments(donors.hours,donors.offsets,donors.items,donors.labels,zero,n,k)
    live=posteriors(n,k,cat,zero,hours,1)
    return frozen,live


@njit
def project(hours,offsets,items,table,start):
    out=np.empty(len(items),dtype=np.float32)
    for i in range(len(hours)):
        a,b=offsets[i:i+2]
        for j in range(a,b):
            out[j]=table[hours[i]-start,items[j]]
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",type=Path,required=True)
    args=ap.parse_args()
    tables=prepare(args.data)
    for domain,name in [("small","MINDsmall_dev.zip"),("full","gain_full_large.zip")]:
        period=read(args.data/name)
        for variant,table in zip(("frozen","live"),tables):
            values=project(period.hours,period.offsets,period.items,np.stack(list(table.values())),min(table))
            np.save(args.data/f"gain_{domain}_{variant}.npy",values)
        print(domain,len(period.ids),len(period.labels),flush=True)
