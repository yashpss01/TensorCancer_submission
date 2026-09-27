"""Train candidate matcher and tune only on development validation data."""
import argparse
import hashlib
import json
import math
import pathlib
import resource
import sys
import time

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT=pathlib.Path(__file__).resolve().parents[3]
R3=ROOT/'code/business_entity_resolution/round3'
R2=ROOT/'code/business_entity_resolution/round2'
sys.path[:0]=[str(R3),str(R2)]
import evaluate_60k as prior
from common import load,dump
from features import FEATURE_NAMES

ART=ROOT/'artifacts/matching_round5'
R4=ROOT/'artifacts/blocking_round4/fresh'


def digest(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def arrays(split):
    path=ART/split
    x=np.load(path/'X.npy',mmap_mode='r')
    y=np.load(path/'y.npy',mmap_mode='r')
    offsets=np.load(path/'offsets.npy')
    meta=load(path/'meta.json')
    assert x.shape==(meta['pairs'],len(FEATURE_NAMES))
    assert y.shape==(meta['pairs'],)
    assert len(offsets)==meta['queries']+1 and offsets[-1]==meta['pairs']
    assert meta['features']==FEATURE_NAMES
    return x,y,offsets,meta


def group_truth(split):
    if split=='validation':
        q=load(R4/'queries.json'); truth=load(R4/'truth.json')
    elif split=='holdout':
        q=load(prior.FRESH/'queries.json')[40000:60000]
        truth=load(prior.FRESH/'truth.json')
    else: raise ValueError(split)
    counts=np.array([len(truth[x['entity_id']]) for x in q],dtype=np.int32)
    india=np.array([x['country']=='India' for x in q],dtype=bool)
    us=np.array([x['country']=='US' for x in q],dtype=bool)
    return counts,dict(India=india,US=us)


def score_prob(prob,y,offsets,truth_counts,country_masks,threshold):
    chosen=prob>=threshold
    pc=np.concatenate(([0],np.cumsum(chosen,dtype=np.int64)))
    hc=np.concatenate(([0],np.cumsum(chosen*y,dtype=np.int64)))
    pred=pc[offsets[1:]]-pc[offsets[:-1]]
    hit=hc[offsets[1:]]-hc[offsets[:-1]]
    fp=pred-hit
    scores=np.where(truth_counts==0,(pred==0).astype(float),1.25*hit/np.maximum(1e-12,hit+.25*truth_counts+fp))
    def aggregate(mask):
        return dict(queries=int(mask.sum()),macro_f05=float(scores[mask].mean()),pair_tp=int(hit[mask].sum()),pair_fp=int(fp[mask].sum()),pair_fn=int((truth_counts[mask]-hit[mask]).sum()),singleton_count=int((truth_counts[mask]==0).sum()),singleton_correct=int(((truth_counts==0)&(pred==0)&mask).sum()),perfect_entities=int(((scores==1)&mask).sum()))
    return dict(overall=aggregate(np.ones(len(scores),dtype=bool)),countries={c:aggregate(m) for c,m in country_masks.items()})


def optimize(prob,y,offsets,truth_counts,country_masks):
    grid=np.unique(np.concatenate((np.linspace(.01,.99,99),np.array([.001,.005,.995,.997,.999,.9995]))))
    out=[dict(threshold=float(th),**score_prob(prob,y,offsets,truth_counts,country_masks,float(th))) for th in grid]
    best=max(out,key=lambda x:(x['overall']['macro_f05'],-x['overall']['pair_fp']))
    return out,best


def train():
    prior.verify_inputs()
    assert load(R4/'validation.json')['status']=='PASS'
    if (ART/'model_results.json').exists(): raise RuntimeError('Already trained')
    x,y,_,mt=arrays('train')
    vx,vy,vo,mv=arrays('validation')
    assert mt['queries']==30000 and mv['queries']==10000
    counts,masks=group_truth('validation')
    start=time.monotonic()
    outputs={}

    linear=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=300,solver='lbfgs'))
    linear.fit(x,y)
    linear_path=ART/'logistic.joblib'
    joblib.dump(linear,linear_path)
    linear_prob=linear.predict_proba(vx)[:,1].astype('float32')
    np.save(ART/'validation_logistic_prob.npy',linear_prob)
    grid,best=optimize(linear_prob,vy,vo,counts,masks)
    dump(ART/'logistic_grid.json',grid)
    outputs['logistic']=dict(best=best,model_path=str(linear_path.relative_to(ROOT)),model_sha256=digest(linear_path),validation_prob_sha256=digest(ART/'validation_logistic_prob.npy'))
    print('logistic',best['threshold'],best['overall']['macro_f05'],flush=True)

    xgb=XGBClassifier(n_estimators=700,max_depth=7,learning_rate=.06,min_child_weight=5,subsample=.9,colsample_bytree=.9,reg_lambda=3,tree_method='hist',n_jobs=8,objective='binary:logistic',eval_metric='logloss',early_stopping_rounds=35,random_state=20260926)
    xgb.fit(x,y,eval_set=[(vx,vy)],verbose=False)
    xgb_path=ART/'xgboost.json'
    xgb.save_model(xgb_path)
    xgb_prob=xgb.predict_proba(vx)[:,1].astype('float32')
    np.save(ART/'validation_xgboost_prob.npy',xgb_prob)
    grid,best=optimize(xgb_prob,vy,vo,counts,masks)
    dump(ART/'xgboost_grid.json',grid)
    outputs['xgboost']=dict(best=best,model_path=str(xgb_path.relative_to(ROOT)),model_sha256=digest(xgb_path),validation_prob_sha256=digest(ART/'validation_xgboost_prob.npy'),best_iteration=int(xgb.best_iteration))
    print('xgboost',best['threshold'],best['overall']['macro_f05'],flush=True)

    dump(ART/'model_results.json',dict(models=outputs,train_rows=mt['pairs'],validation_rows=mv['pairs'],seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,feature_names=FEATURE_NAMES))
    print('Training complete',round(time.monotonic()-start,1),flush=True)


def freeze():
    prior.verify_inputs()
    if (ART/'frozen_model.json').exists(): raise RuntimeError('Already frozen')
    results=load(ART/'model_results.json')
    best_name=max(results['models'],key=lambda m:results['models'][m]['best']['overall']['macro_f05'])
    best=results['models'][best_name]
    files=[ROOT/'code/business_entity_resolution/round5/features.py',ROOT/'code/business_entity_resolution/round5/dataset.py',pathlib.Path(__file__)]
    files += [ROOT/'code/business_entity_resolution/round2/normalize.py',ROOT/'code/business_entity_resolution/round2/improve.py']
    cfg=dict(model=best_name,threshold=best['best']['threshold'],development_macro_f05=best['best']['overall']['macro_f05'],development_india_f05=best['best']['countries']['India']['macro_f05'],model_path=best['model_path'],model_sha256=best['model_sha256'],code_hashes={str(p.relative_to(ROOT)):digest(p) for p in files},train_meta_sha256=digest(ART/'train/meta.json'),validation_meta_sha256=digest(ART/'validation/meta.json'),validation_results_sha256=digest(ART/'model_results.json'),frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),holdout='Original S1 rows 60001–80000 on the sealed 409141-target pool; no holdout labels used for model selection')
    dump(ART/'frozen_model.json',cfg)
    print('Frozen',best_name,best['best']['threshold'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','freeze']);a=p.parse_args()
    globals()[a.stage]()
