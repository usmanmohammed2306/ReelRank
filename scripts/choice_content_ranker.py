"""Content- and exposure-aware ranking on complete SM-RS recommendation rounds."""
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

from scripts import choice_fusion as fusion
from scripts.choice_models import percentile
from smrs.tasks.task3_reranking import evaluate

GOODBOOKS_REV = "6dd165b555a7b47b2dd36743a425776e641ff50c"
HASHES = {
    "goodbooks-meta.csv":"4d21d0e0433128657c752e3a3d0405ac0ec11675d6c70c64cb0abf4ab1acb76a",
    "goodbooks-tags.csv":"8ff2f0659be9ddbe10bf47ba33b5b69024f893f017ac665236f15bc97d862950"}
MOVIE_MD5 = "6b51fb2759a8657d3bfcbfc42b592ada"


def digest(path, algorithm="sha256"):
    h=hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_in(z, name, **kwargs):
    return pd.read_csv(z.open("ml-25m/"+name+".csv"),**kwargs)


def low_rank(x):
    return normalize(TruncatedSVD(n_components=32,n_iter=4,random_state=575).fit_transform(x))


def centered(x):
    x=np.asarray(x,dtype=float)
    return x-x.mean(axis=0)


@dataclass
class Catalog:
    offset: int
    qualities: np.ndarray
    counts: np.ndarray
    years: np.ndarray
    tags: np.ndarray
    titles: csr_matrix
    creators: csr_matrix
    ease: np.ndarray
    popularity: np.ndarray


def catalog(root, movies_zip):
    if digest(movies_zip,"md5")!=MOVIE_MD5:
        raise ValueError("MovieLens 25M archive digest differs from the official release")
    for name,checksum in HASHES.items():
        if digest(root/name)!=checksum:
            raise ValueError("Unexpected Goodbooks source bytes: "+name)
    maps=pd.read_parquet(root/"data/movies.parquet").sort_values("movieId")
    bmap=pd.read_parquet(root/"data/books.parquet").sort_values("book_index")
    with ZipFile(movies_zip) as z:
        meta=csv_in(z,"movies").set_index("movieId")
        links=csv_in(z,"links")
        movie_ids=maps.imdbId.map(dict(zip(links.imdbId,links.movieId))).fillna(-1).astype(int).to_numpy()
        meta=meta.reindex(movie_ids)
        years=pd.to_numeric(meta.title.str.extract(r'\((\d{4})\)\s*$',expand=False),errors="coerce").to_numpy()
        titles=meta.title.fillna("").str.replace(r'\(\d{4}\)\s*$','',regex=True).to_list()
        creators=meta.genres.fillna("").str.lower().to_list()
        counts=np.zeros((max(links.movieId)+1,4))
        for chunk in csv_in(z,"ratings",usecols=["movieId","rating"],chunksize=1_000_000):
            ids=chunk.movieId.to_numpy(); y=chunk.rating.to_numpy()
            for col,w in enumerate((None,y,(y>=4).astype(float),(y==5).astype(float))):
                counts[:,col]+=np.bincount(ids,weights=w,minlength=len(counts))
        stars=counts[np.maximum(0,movie_ids)]
        known=stars[:,0]
        stats=np.column_stack([(stars[:,1]+3.5*10)/(known+10),(stars[:,2]+5)/(known+10),(stars[:,3]+2)/(known+10)])
        genome=csv_in(z,"genome-scores",dtype={"movieId":"int32","tagId":"int16","relevance":"float32"})
        gids,inv=np.unique(genome.movieId,return_inverse=True)
        matrix=np.zeros((len(gids),int(genome.tagId.max())),dtype=np.float32)
        matrix[inv,genome.tagId.to_numpy()-1]=genome.relevance
        del genome
        matrix-=matrix.mean(axis=0)
        basis=low_rank(matrix)
        movie_tags=np.zeros((len(movie_ids),32),dtype=np.float32)
        lookup=pd.Index(gids).get_indexer(movie_ids); ok=lookup>=0
        movie_tags[ok]=basis[lookup[ok]]
    books=pd.read_csv(root/"goodbooks-meta.csv").drop_duplicates("goodreads_book_id").set_index("goodreads_book_id").reindex(bmap.goodreads_id)
    book_counts=books.ratings_count.fillna(0).to_numpy()
    bstats=np.column_stack([books.average_rating.fillna(3.7),(books.ratings_4+books.ratings_5).fillna(0)/(book_counts+1),books.ratings_5.fillna(0)/(book_counts+1)])
    btags=pd.read_csv(root/"goodbooks-tags.csv")
    idx=pd.Index(bmap.goodreads_id).get_indexer(btags.goodreads_book_id); keep=(idx>=0)&(btags["count"]>0)
    bt=csr_matrix((np.log1p(btags.loc[keep,"count"].to_numpy(dtype=float)),(idx[keep],btags.loc[keep,"tag_id"].to_numpy())),
                  shape=(len(books),btags.tag_id.max()+1),dtype=np.float32)
    bt=low_rank(TfidfTransformer().fit_transform(bt))
    domains={"movies":(0,titles,creators,years,known,stats,movie_tags),
             "books":(len(maps),books.title.fillna("").to_list(),books.authors.fillna("").str.lower().str.replace(', ','|',regex=False).to_list(),
                      books.original_publication_year.to_numpy(),book_counts,bstats,bt)}
    result={}
    for domain,(offset,ti,cr,yr,num,quality,tags) in domains.items():
        pop=np.load(root/"artifacts"/f"mean_popularities_{domain}.npy")
        ease=np.load(root/"artifacts"/f"full_{domain}_item_item.npy",mmap_mode="r")
        lex=TfidfVectorizer(strip_accents="unicode",stop_words="english",ngram_range=(1,2),dtype=np.float32).fit_transform(ti)
        categories=TfidfVectorizer(tokenizer=lambda s:[v.strip() for v in s.split("|") if v.strip()],token_pattern=None,
                                   dtype=np.float32).fit_transform(cr)
        yr=np.nan_to_num(yr,nan=1985.).clip(1800,2024)
        # A separate domain CDF makes public ratings-count and rating-distribution scales comparable.
        qual=np.column_stack([percentile(col) for col in quality.T]+[percentile(num), (yr-1980)/30,
                             percentile(yr),np.log1p(num)/12, np.asarray([len(t.split())/6 for t in ti])])
        result[domain]=Catalog(offset,qual,num,yr,tags,lex,categories,ease,pop)
    return result


def dated_history(log):
    """Expose titles when displayed and clicks only after both display and click time."""
    views, clicks=[],[]
    for row in log.itertuples():
        start=row.timestamp.value/1e9
        views.append((start,{int(x["itemId"]) for x in row.impressions}))
        for x in row.impressions:
            if x["selected"] and np.isfinite(x["timestamp"]):
                clicks.append((max(start,float(x["timestamp"])),int(x["itemId"])))
    clicks.sort();views.sort(key=lambda x:x[0])
    profile, exposure, click_counts={},Counter(),Counter()
    ci=vi=0
    for row in log.loc[log.iteration.notna()].sort_values(["timestamp","impressionId"]).itertuples():
        now=row.timestamp.value/1e9
        while ci<len(clicks) and clicks[ci][0]<now:
            it=clicks[ci][1];profile.pop(it,None);profile[it]=None;click_counts[it]+=1;ci+=1
        while vi<len(views) and views[vi][0]<now:
            exposure.update(views[vi][1]);vi+=1
        yield row,np.fromiter(profile,dtype=int),exposure,click_counts


def similarities(vectors, ids, prior):
    if not len(prior):
        return np.zeros((len(ids),0))
    val=vectors[ids]@vectors[prior].T
    return val.toarray() if hasattr(val,"toarray") else np.asarray(val)


def signals(cat, items, hist, exposures, clicks):
    ix=items-cat.offset; h=hist-cat.offset
    shown=np.fromiter(exposures.keys(),dtype=int)-cat.offset
    cols=[cat.qualities[ix],centered(cat.qualities[ix]),(cat.years[ix]>2000)[:,None],
          (cat.years[ix]>2010)[:,None]]
    if len(h):
        dist=np.abs(cat.years[ix,None]-cat.years[h])
        cols.append(np.column_stack([dist.mean(1)/25,dist.min(1)/10,(dist<6).mean(1)]))
    else:
        cols.append(np.zeros((len(ix),3)))
    for vec in (cat.tags,cat.creators,cat.titles):
        pos=similarities(vec,ix,h)
        offered=similarities(vec,ix,shown)
        if pos.shape[1]:
            near=np.partition(pos,max(0,pos.shape[1]-3),axis=1)[:,-3:].mean(axis=1)
            sp=np.column_stack([pos.mean(1),pos.max(1),near,pos[:,-5:].mean(1)])
        else:
            sp=np.zeros((len(ix),4))
        base=offered.mean(axis=1) if len(shown) else np.zeros(len(ix))
        cols.extend([sp, (sp[:,0]-base)[:,None],centered(sp)])
    # Keep absolute and within-list weighted collaborative affinity as well as ordinal expert scores.
    e=cat.ease[np.ix_(h,ix)] if len(h) else np.zeros((1,len(ix)))
    raw=np.column_stack([np.log1p(1000*cat.popularity[ix]),
                         np.log1p([exposures[int(i)] for i in items]), np.log1p([clicks[int(i)] for i in items]),
                         e.sum(0),e.max(0),e[-5:].mean(0),np.partition(e,max(0,len(e)-5),axis=0)[-5:].mean(0)])
    cf=np.column_stack([raw,centered(raw)]+[percentile(raw[:,i]) for i in range(raw.shape[1])])
    return np.column_stack(cols),cf


@dataclass
class Example:
    user:int
    slate:int
    domain:str
    ids:np.ndarray
    y:np.ndarray
    fusion:np.ndarray
    cf:np.ndarray
    content:np.ndarray
    pop_bins:np.ndarray


def examples(root, catalogs, users):
    logs=pd.read_parquet(root/"data/behaviors.parquet",filters=[("userId","in",list(map(int,users)))])
    domains=pd.read_parquet(root/"data/users.parquet").set_index("userId").domain
    orders=fusion.read_orders(root)
    bins={d:np.minimum((20*percentile(c.popularity)).astype(int),19)+20*(d=="books") for d,c in catalogs.items()}
    data=[]
    for uid,group in logs.groupby("userId"):
        d=domains[uid];cat=catalogs[d]
        for row,h,ex,cl in dated_history(group):
            ids=np.array([int(x["itemId"]) for x in row.impressions])
            ranks=fusion.candidate_ranks(ids,[orders[m][uid,row.impressionId] for m in fusion.EXPERTS])
            features,bounds=fusion.design(ranks,d,len(h),np.zeros(6),np.zeros(6))
            ctx=features[:,:bounds[1]]
            y=np.array([bool(o["selected"]) for o in row.impressions])
            content,cf=signals(cat,ids,h,ex,cl)
            data.append(Example(int(uid),int(row.impressionId),d,ids,y,ctx,cf,content,bins[d][ids-cat.offset]))
    return data


class Uptake:
    """Conditional item rates with every target participant removed from their training encodings."""
    def __init__(self, train):
        self.total=np.zeros((22424,2))
        self.bins=np.zeros((40,2))
        self.people={}
        for c in train:
            if c.user not in self.people:
                self.people[c.user]=({},np.zeros((40,2)))
            items,bins=self.people[c.user]
            for i,b,positive in zip(c.ids,c.pop_bins,c.y):
                delta=np.array([int(positive),1])
                self.total[i]+=delta; self.bins[b]+=delta;bins[b]+=delta
                items[i]=items.get(i,0)+delta

    def transform(self, c):
        observed=self.total[c.ids].copy()
        group=self.bins[c.pop_bins].copy()
        if c.user in self.people:
            counts,buckets=self.people[c.user]
            for n,i in enumerate(c.ids):
                observed[n]-=counts.get(i,0)
            group-=buckets[c.pop_bins]
        base=(group[:,0]+5)/(group[:,1]+15)
        k,n=observed.T
        raw=np.column_stack([(k+10*base)/(n+10),(k+40*base)/(n+40),
                             (k+3)/(n+10),np.log1p(n),base])
        return np.column_stack([raw,centered(raw)])


def concatenate(cases, bank):
    columns=[]
    for c in cases:
        tab=bank.transform(c)
        signed=1 if c.domain=="books" else -1
        extras=np.column_stack([c.cf,tab])
        extra_content=np.column_stack([c.content,signed*c.content])
        columns.append(np.column_stack([c.fusion,extras,signed*extras,extra_content]))
    return np.stack(columns)


def rows_for(test, predictions):
    out=[]
    for method, scores in predictions.items():
        for c,v in zip(test,scores):
            order=np.lexsort((fusion.tie_key(c.ids,c.slate),-v))
            y=c.y[order]
            first=np.flatnonzero(y)
            discount=1/np.log2(np.arange(2,12))
            den=discount[:int(y.sum())].sum()
            out.append((c.user,c.slate,c.domain,method,",".join(map(str,c.ids[order])),
                        float((y@discount)/den) if den else 0.,y[:5].mean(),int(y[0]),int(y[:5].any()),
                        int(first[0]+1) if len(first) else 11))
    return pd.DataFrame(out,columns=["user","slate","domain","method","ranking","ndcg","p5","hit1","hit5","first"])


def report(rows):
    metrics=list(fusion.METRICS)
    def m(g):return g.groupby("method")[metrics].mean().to_dict(orient="index")
    return {"participants":int(rows.user.nunique()),"slates":int(rows.slate.nunique()),"metrics":m(rows),
            "domains":{d:{"participants":int(g.user.nunique()),"slates":int(g.slate.nunique()),"metrics":m(g)}
                       for d,g in rows.groupby("domain")}}


def canonical(cases, rows):
    target={c.slate:set(c.ids[c.y]) for c in cases}
    for _,g in rows.groupby("method"):
        ranks=[[int(x) for x in s.split(",")] for s in g.ranking]
        scores=evaluate(ranks,[target[i] for i in g.slate])
        np.testing.assert_allclose([scores["nDCG@10"],scores["Precision@5"]],[g.ndcg.mean(),g.p5.mean()],atol=1e-12)
