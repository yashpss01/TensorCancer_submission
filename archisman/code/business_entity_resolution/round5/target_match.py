"""Learn S2/S3 variant consistency from labeled training groups."""
import argparse
import hashlib
import itertools
import json
import pathlib
import resource
import sys
import time

import numpy as np
from xgboost import XGBClassifier

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import dataset
import train as first
from features import FEATURE_NAMES,features,represent
from improve import select
from common import load,dump

ART=ROOT/'artifacts/matching_round5'


def choices(true,mids,probs):
    true=sorted(set(true))
    if not true: return [],[]
    anchors=sorted(true,key=lambda x:hashlib.sha256(x.encode()).digest())[:2]
    neg=[(float(p),mid) for p,mid in zip(probs,mids) if mid not in set(true)]
    neg.sort(key=lambda x:(-x[0],x[1]))
    chosen=[mid for _,mid in neg[:12]]
    tail=neg[12:]
    if tail:
        for i in [len(tail)//4,len(tail)//2,3*len(tail)//4]:
            mid=tail[i][1]
            if mid not in chosen: chosen.append(mid)
    return list(itertools.combinations(true,2)),[(a,mid) for a in anchors for mid in chosen]


def build():
    out=ART/'target_pairs'
    if out.exists(): raise RuntimeError('Target-pair data exists')
    out.mkdir()
    queries=load(dataset.prior.FRESH/'queries.json')[:30000]
    truth=load(dataset.prior.FRESH/'truth.json')
    prob=np.load(ART/'train_oof_prob.npy',mmap_mode='r')
    offsets=np.load(ART/'train/offsets.npy')
    counts=[]
    for gi,(q,r) in enumerate(zip(queries,dataset.read_jsonl(dataset.prior.FRESH/'improved.jsonl'))):
        mids=select(r,.5,16)
        positive,negative=choices(truth[q['entity_id']],mids,prob[offsets[gi]:offsets[gi+1]])
        counts.append((len(positive),len(negative)))
    assert len(counts)==30000
    total=sum(a+b for a,b in counts)
    X=np.lib.format.open_memmap(out/'X.npy',mode='w+',dtype='float32',shape=(total,len(FEATURE_NAMES)))
    y=np.lib.format.open_memmap(out/'y.npy',mode='w+',dtype='uint8',shape=(total,))
    folds=np.lib.format.open_memmap(out/'fold.npy',mode='w+',dtype='uint8',shape=(total,))
    records=dataset.load_records();nw,aw,default=dataset.load_weights()
    index=0;start=time.monotonic()
    for gi,(q,r) in enumerate(zip(queries,dataset.read_jsonl(dataset.prior.FRESH/'improved.jsonl')),1):
        mids=select(r,.5,16)
        positive,negative=choices(truth[q['entity_id']],mids,prob[offsets[gi-1]:offsets[gi]])
        fold=int.from_bytes(hashlib.sha256(q['entity_id'].encode()).digest()[:4],'big')%5
        for label,pairs in [(1,positive),(0,negative)]:
            for a,b in pairs:
                X[index]=features(represent(*records[a]),represent(*records[b]),nw,aw,default)
                y[index]=label
                folds[index]=fold
                index+=1
        if gi%1000==0:
            X.flush();y.flush();folds.flush()
            print('target pairs',gi,index,round(time.monotonic()-start,1),flush=True)
    assert index==total
    X.flush();y.flush();folds.flush()
    dump(out/'meta.json',dict(groups=30000,pairs=total,positives=int(y.sum()),seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,features=FEATURE_NAMES))


def train():
    out=ART/'target_model.json'
    if out.exists():raise RuntimeError('Target model exists')
    path=ART/'target_pairs'
    X=np.load(path/'X.npy',mmap_mode='r');y=np.load(path/'y.npy',mmap_mode='r');fold=np.load(path/'fold.npy',mmap_mode='r')
    train_mask=fold!=0
    model=XGBClassifier(n_estimators=800,max_depth=7,learning_rate=.06,min_child_weight=5,subsample=.9,colsample_bytree=.9,reg_lambda=3,tree_method='hist',n_jobs=8,objective='binary:logistic',eval_metric='logloss',early_stopping_rounds=40,random_state=20260929)
    start=time.monotonic()
    model.fit(X[train_mask],y[train_mask],eval_set=[(X[~train_mask],y[~train_mask])],verbose=False)
    model.save_model(out)
    p=model.predict_proba(X[~train_mask])[:,1]
    dump(ART/'target_model_meta.json',dict(best_iteration=int(model.best_iteration),seconds=time.monotonic()-start,positive_mean_probability=float(p[y[~train_mask]==1].mean()),negative_mean_probability=float(p[y[~train_mask]==0].mean()),validation_pairs=int((~train_mask).sum()),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print('target model',model.best_iteration,round(time.monotonic()-start,1),flush=True)


def predict_validation():
    path=ART/'validation_target_match_max.npy'
    if path.exists(): raise RuntimeError('Target-match scores exist')
    model=XGBClassifier();model.load_model(ART/'target_model.json')
    queries=load(dataset.R4/'queries.json')
    raw=dataset.read_jsonl(dataset.R4/'improved.jsonl')
    offsets=np.load(ART/'validation/offsets.npy')
    pair_prob=np.load(ART/'validation_xgboost_prob.npy',mmap_mode='r')
    out=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(pair_prob),))
    out[:]=0
    records=dataset.load_records();nw,aw,default=dataset.load_weights()
    batch=[];positions=[];start=time.monotonic()
    def flush():
        if not batch:return
        p=model.predict_proba(np.asarray(batch,dtype='float32'))[:,1]
        for index,value in zip(positions,p):
            if value>out[index]:out[index]=value
        batch.clear();positions.clear()
    for gi,(q,r) in enumerate(zip(queries,raw),1):
        mids=select(r,.5,16)
        lo,hi=int(offsets[gi-1]),int(offsets[gi])
        assert len(mids)==hi-lo
        ps=np.asarray(pair_prob[lo:hi]);order=np.argsort(-ps,kind='stable')
        anchors=[int(j) for j in order[:5] if ps[j]>=.9][:3]
        if not anchors and len(order):anchors=[int(order[0])]
        reps=[represent(*records[mid]) for mid in mids]
        for j,rep in enumerate(reps):
            for a in anchors:
                if j==a:continue
                batch.append(features(rep,reps[a],nw,aw,default))
                positions.append(lo+j)
                if len(batch)>=20000:flush()
        if gi%1000==0:
            flush();out.flush()
            print('target max',gi,round(time.monotonic()-start,1),flush=True)
    flush();out.flush()
    dump(ART/'validation_target_match_meta.json',dict(rows=len(out),seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


def tune():
    prob=np.load(ART/'validation_xgboost_prob.npy')
    target=np.load(ART/'validation_target_match_max.npy')
    y=np.load(ART/'validation/y.npy',mmap_mode='r')
    offsets=np.load(ART/'validation/offsets.npy')
    counts,masks=first.group_truth('validation')
    results=[]
    for base in [.74,.8,.85,.9]:
        for floor in [.01,.05,.1,.2,.3,.4,.5,.6]:
            for threshold in [.7,.8,.9,.95,.98,.99,.995]:
                chosen=(prob>=base)|((prob>=floor)&(target>=threshold))
                score=first.score_prob(chosen.astype('float32'),y,offsets,counts,masks,.5)
                results.append(dict(base_threshold=base,pair_floor=floor,target_threshold=threshold,**score))
    best=max(results,key=lambda z:z['overall']['macro_f05'])
    dump(ART/'target_rescue_grid.json',results)
    dump(ART/'target_rescue_best.json',best)
    print('best target rescue',best['base_threshold'],best['pair_floor'],best['target_threshold'],best['overall']['macro_f05'],best['countries']['India']['macro_f05'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','train','predict_validation','tune']);a=p.parse_args()
    globals()[a.stage]()
