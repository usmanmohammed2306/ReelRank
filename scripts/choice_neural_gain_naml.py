"""Score whole MIND lists with the three published NewsReX NAML checkpoints."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from safetensors.torch import load_file

from choice_external_nrms import CachedArticles

ROOT = Path(__file__).resolve().parents[1]


def all_large(data):
    """Remap all published large lists to the verified article-URL crosswalk."""
    path = data / "gain_full_large.zip"
    with ZipFile(data / "large_cohort.zip") as canonical, ZipFile(data / "MINDlarge_dev.zip") as raw, ZipFile(path,"w") as dst:
        metadata = canonical.read("news.tsv")
        dst.writestr("news.tsv",metadata)
        urls = {r.split(b"\t")[5]:r.split(b"\t")[0] for r in metadata.splitlines()}
        ids = {}
        for r in raw.open("news.tsv"):
            parts = r.split(b"\t")
            ids[parts[0]]=urls[parts[5]]
        with dst.open("behaviors.tsv","w") as writer:
            for r in raw.open("behaviors.tsv"):
                sid,user,dt,hist,items=r.rstrip(b"\r\n").split(b"\t")
                hist=b" ".join(ids[k] for k in hist.split())
                items=b" ".join(ids[q[:-2]]+q[-2:] for q in items.split())
                writer.write(b"\t".join((sid,user,dt,hist,items))+b"\n")
    return path


def packed_data(data, target):
    from src.core.data.processing.text.news import read_all_news
    from src.core.data.processing.text.vocabulary import build_vocabulary, tokenize_text
    news=read_all_news(data)
    vocab=build_vocabulary(news,word_threshold=5)
    categories={s:i for i,s in enumerate(sorted(news.category.unique()))}
    subcategories={s:i for i,s in enumerate(sorted(news.subcategory.unique()))}
    records={int(r.id[1:]):(str(r.title),str(r.abstract),r.category,r.subcategory) for r in news.itertuples()}
    with ZipFile(target) as z:
        for r in z.open("news.tsv"):
            item,cat,sub,title,abst,*_=r.decode().rstrip("\r\n").split("\t")
            records.setdefault(int(item[1:]),(title,abst,cat,sub))
    ids=np.asarray(list(records),dtype=np.int64)
    title=np.zeros((ids.max()+1,32),dtype=np.int64)
    abstract=np.zeros((len(title),50),dtype=np.int64)
    index=np.zeros((len(title),3),dtype=np.int64)
    for k,(t,a,c,s) in records.items():
        title[k]=tokenize_text(t,vocab,32,1,0)
        abstract[k]=tokenize_text(a,vocab,50,1,0)
        index[k]=[k,categories.get(c,0),subcategories.get(s,0)]
    return title,abstract,torch.from_numpy(index),torch.from_numpy(ids),len(vocab),len(categories),len(subcategories)


def score_lists(user,vectors,archive,out):
    rows={b"":0}; ix=[]
    with ZipFile(archive) as z:
        for r in z.open("behaviors.tsv"):
            h=r.split(b"\t",4)[3]
            ix.append(rows.setdefault(h,len(rows)))
    histories=torch.zeros((len(rows),50),dtype=torch.long)
    for h,k in rows.items():
        v=[int(i[1:]) for i in h.split()[-50:]]
        if v: histories[k,-len(v):]=torch.tensor(v)
    uc=torch.cat([user(batch,training=False) for batch in histories.split(128)]).numpy()
    a=vectors.numpy()
    scores=[]
    with ZipFile(archive) as z:
        for j,r in enumerate(z.open("behaviors.tsv")):
            ids=[int(s[1:-2]) for s in r.rstrip(b"\r\n").split(b"\t",4)[4].split()]
            scores.append(a[ids] @ uc[ix[j]])
    np.save(out,np.concatenate(scores))


class CachedTokenConvolution(torch.nn.Module):
    """Exact inference convolution from cached per-vocabulary kernel projections."""
    def __init__(self, view):
        super().__init__()
        kernel=view.cnn.weight
        if kernel.shape[2]!=3:
            raise ValueError("Expected the published three-token NAML convolution")
        self.word_ids=view.text_encoder.word_ids
        self.projections=[view.text_encoder.embedding.weight @ kernel[:,:,j].T for j in range(3)]
        self.bias=view.cnn.bias
        self.attention=view.additive_attention
        self.activation=view.activation

    def forward(self, news_idx, training=False):
        if training:
            raise ValueError("Convolution caching is for inference only")
        ids=self.word_ids[news_idx]
        out=self.projections[1][ids]+self.bias
        out[:,1:]+=self.projections[0][ids[:,:-1]]
        out[:,:-1]+=self.projections[2][ids[:,1:]]
        return self.attention(self.activation(out),mask=ids!=0)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--upstream",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[42,123,456])
    p.add_argument("--threads",type=int,default=3)
    p.add_argument("--cache-convolutions",action="store_true")
    opts=p.parse_args()
    sys.path.insert(0,str(opts.upstream))
    from src.frameworks.pytorch.layers import GloveTextEncoder
    from src.frameworks.pytorch.models.naml import NewsEncoder, UserEncoder
    from src.core.models.configs import NAMLConfig
    torch.set_num_threads(opts.threads)
    prot=json.loads((ROOT/"results/choice_gains/mind_inputs.json").read_text())
    full=opts.data/"gain_full_large.zip" if opts.cache_convolutions and (opts.data/"gain_full_large.zip").exists() else all_large(opts.data)
    for part,target in (("train","train"),("dev","test")):
        with ZipFile(opts.data/f"MINDsmall_{part}.zip") as z:
            z.extract("news.tsv",opts.data/target)
    title,abstract,ix,ids,vocab,cat,sub=packed_data(opts.data,full)
    for seed in opts.seeds:
        file=opts.data/f"NAML_{seed}_model.safetensors"
        if hashlib.sha256(file.read_bytes()).hexdigest()!=prot["naml_checkpoints"][str(seed)]:
            raise ValueError("Unexpected NAML checkpoint")
        weights=load_file(str(file))
        cfg=NAMLConfig(max_title_length=32,max_abstract_length=50,seed=seed)
        d=300
        enc=NewsEncoder(cfg,GloveTextEncoder(title,vocab_size=vocab,embedding_dim=d),
                        GloveTextEncoder(abstract,vocab_size=vocab,embedding_dim=d),cat,sub)
        user=UserEncoder(cfg,enc)
        for module,name in [(enc,"news_encoder."),(user,"user_encoder.")]:
            module.load_state_dict({k[len(name):]:v for k,v in weights.items() if k.startswith(name)},strict=True)
            module.eval()
        if opts.cache_convolutions:
            sample=ids[::max(1,len(ids)//128)][:128]
            original=enc(ix[sample],training=False)
            enc.title_view=CachedTokenConvolution(enc.title_view)
            enc.abstract_view=CachedTokenConvolution(enc.abstract_view)
            cached=enc(ix[sample],training=False)
            torch.testing.assert_close(cached,original,rtol=2e-5,atol=2e-5)
            print("Original vs cached max abs error",seed,float((original-cached).abs().max()),flush=True)
        vec=torch.zeros(len(title),cfg.cnn_filter_num)
        for batch in ids.split(128):
            vec[batch]=enc(ix[batch],training=False)
        print("Finished article embeddings",seed,flush=True)
        user.news_encoder=CachedArticles(vec)
        for cohort,path in [("small",opts.data/"MINDsmall_dev.zip"),("full",full)]:
            score_lists(user,vec,path,opts.data/f"gain_{cohort}_naml_{seed}{'_cached' if opts.cache_convolutions else ''}.npy")
            print("Scored",seed,cohort,flush=True)


if __name__=="__main__":
    main()
