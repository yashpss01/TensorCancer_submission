"""Build supervised candidate-pair arrays from sealed blocking outputs."""
import argparse
import hashlib
import json
import math
import pathlib
import resource
import sqlite3
import sys
import time

import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[3]
R2=ROOT/'code/business_entity_resolution/round2'
R3=ROOT/'code/business_entity_resolution/round3'
sys.path[:0]=[str(R3),str(R2)]
import evaluate_60k as prior
from common import load,dump
from improve import select
from features import FEATURE_NAMES,represent,features

ART=ROOT/'artifacts/matching_round5'
R4=ROOT/'artifacts/blocking_round4/fresh'
THRESHOLD=.5
CAP=16


def digest(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def sources(split):
    if split=='train':
        q=load(prior.FRESH/'queries.json')[:30000]
        return q,prior.FRESH/'improved.jsonl',load(prior.FRESH/'truth.json'),30000
    if split=='validation':
        q=load(R4/'queries.json')
        return q,R4/'improved.jsonl',load(R4/'truth.json'),10000
    if split=='holdout':
        q=load(prior.FRESH/'queries.json')[40000:60000]
        return q,ART/'holdout/improved.jsonl',load(prior.FRESH/'truth.json'),20000
    raise ValueError(split)


def read_jsonl(path):
    with open(path) as f:
        for line in f: yield json.loads(line)


def load_records():
    db=sqlite3.connect(f'file:{prior.FRESH/"index.sqlite"}?mode=ro',uri=True)
    return {id:(name,addr,country) for id,name,addr,country in db.execute('SELECT id,name,address,country FROM records')}


def load_weights():
    db=sqlite3.connect(f'file:{prior.FRESH/"extra.sqlite"}?mode=ro',uri=True)
    weights={'native':{},'addr':{}}
    n=load(prior.FRESH/'index_meta.json')['target_count']
    for term,col,doc in db.execute("SELECT term,col,doc FROM vocab WHERE col IN ('native','addr')"):
        weights[col][term]=math.log1p(n/(1+doc))
    return weights['native'],weights['addr'],math.log1p(n)


def build(split,limit=None):
    prior.verify_inputs()
    assert load(R4/'validation.json')['status']=='PASS'
    queries,raw,truth,expected=sources(split)
    if split=='holdout': assert (ART/'holdout/frozen_model.json').exists(), 'Freeze model before holdout features'
    if limit: queries=queries[:limit]
    else: assert len(queries)==expected
    out=ART/(f'{split}_pilot{limit}' if limit else split)
    if out.exists():
        if split!='holdout' or {p.name for p in out.iterdir() if p.name in {'X.npy','y.npy','offsets.npy','meta.json'}}:
            raise RuntimeError(f'{out} already exists')
    else:
        out.mkdir(parents=True)
    start=time.monotonic()
    counts=[]
    for q,r in zip(queries,read_jsonl(raw)):
        assert r['id']==q['entity_id']
        counts.append(len(select(r,THRESHOLD,CAP)))
    assert len(counts)==len(queries)
    offsets=np.zeros(len(queries)+1,dtype=np.int64)
    offsets[1:]=np.cumsum(counts)
    total=int(offsets[-1])
    X=np.lib.format.open_memmap(out/'X.npy',mode='w+',dtype='float32',shape=(total,len(FEATURE_NAMES)))
    y=np.lib.format.open_memmap(out/'y.npy',mode='w+',dtype='uint8',shape=(total,))
    records=load_records()
    nw,aw,default=load_weights()
    index=0
    for i,(q,r) in enumerate(zip(queries,read_jsonl(raw)),1):
        assert r['id']==q['entity_id']
        query=represent(q['business_name'],q['business_address'],q['country'])
        true=set(truth[q['entity_id']])
        for mid in select(r,THRESHOLD,CAP):
            target=represent(*records[mid])
            X[index]=features(query,target,nw,aw,default)
            y[index]=mid in true
            index+=1
        if i%1000==0:
            X.flush(); y.flush()
            print(split,i,total,round(time.monotonic()-start,1),flush=True)
    assert index==total
    X.flush(); y.flush()
    np.save(out/'offsets.npy',offsets)
    positives=int(y.sum())
    dump(out/'meta.json',dict(split=split,queries=len(queries),pairs=total,positives=positives,known_true_links=sum(len(truth[q['entity_id']]) for q in queries),singletons=sum(not truth[q['entity_id']] for q in queries),features=FEATURE_NAMES,seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_jsonl=str(raw.relative_to(ROOT)),source_sha256=digest(raw),target_index_sha256=digest(prior.FRESH/'index.sqlite'),threshold=THRESHOLD,cap=CAP))
    if split=='validation' and not limit:
        assert positives==load(R4/'updated/metrics.json')['overall']['TP']
    print('Built',split,len(queries),total,positives,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('split',choices=['train','validation','holdout']); p.add_argument('--limit',type=int); a=p.parse_args()
    build(a.split,a.limit)
