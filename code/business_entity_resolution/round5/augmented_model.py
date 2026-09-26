"""Train a pair matcher with disjoint supplemental positive examples."""
import argparse
import pathlib
import resource
import sys
import time

import numpy as np
from xgboost import XGBClassifier

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import train as first
from common import dump,load

ART=ROOT/'artifacts/matching_round5'


def run(weight):
    tag=f'augmented_{str(weight).replace(".", "p")}'
    result=ART/f'{tag}_results.json'
    if result.exists(): raise RuntimeError('Model exists')
    sup=ART/'supplement'
    sm=load(sup/'meta.json')
    assert sm['no_shared_labeled_target_with_holdout']
    X,y,_,_=first.arrays('train')
    VX,VY,offsets,_=first.arrays('validation')
    E=np.load(ART/'train/extra_pair.npy',mmap_mode='r')
    VE=np.load(ART/'validation/extra_pair.npy',mmap_mode='r')
    SX=np.load(sup/'X.npy',mmap_mode='r')
    SE=np.load(sup/'E.npy',mmap_mode='r')
    assert len(SX)==len(SE)==sm['positive_pairs']
    assert len(X)==len(E)==len(y) and len(VX)==len(VE)==len(VY)
    start=time.monotonic()
    tx=np.empty((len(X)+len(SX),X.shape[1]+E.shape[1]),dtype='float32')
    tx[:len(X),:X.shape[1]]=X
    tx[:len(X),X.shape[1]:]=E
    tx[len(X):,:X.shape[1]]=SX
    tx[len(X):,X.shape[1]:]=SE
    ty=np.empty(len(tx),dtype='uint8')
    ty[:len(y)]=y;ty[len(y):]=1
    weights=np.ones(len(tx),dtype='float32')
    weights[len(y):]=weight
    vx=np.concatenate((VX,VE),axis=1)
    model=XGBClassifier(n_estimators=1500,max_depth=8,learning_rate=.05,min_child_weight=3,subsample=.9,colsample_bytree=.9,reg_lambda=2,tree_method='hist',n_jobs=8,objective='binary:logistic',eval_metric='logloss',early_stopping_rounds=60,random_state=20261001)
    model.fit(tx,ty,sample_weight=weights,eval_set=[(vx,VY)],verbose=False)
    path=ART/f'{tag}.json';model.save_model(path)
    p=model.predict_proba(vx)[:,1].astype('float32')
    np.save(ART/f'validation_{tag}_prob.npy',p)
    counts,masks=first.group_truth('validation')
    grid,best=first.optimize(p,VY,offsets,counts,masks)
    dump(ART/f'{tag}_grid.json',grid)
    dump(result,dict(best=best,best_iteration=int(model.best_iteration),seconds=time.monotonic()-start,features=tx.shape[1],supplement_weight=weight,supplement_pairs=len(SX),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,model=str(path.relative_to(ROOT))))
    print(tag,best['threshold'],best['overall']['macro_f05'],best['countries']['India']['macro_f05'],round(time.monotonic()-start,1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weight',type=float,default=.15);a=p.parse_args()
    if not 0<a.weight<=1: raise ValueError(a.weight)
    run(a.weight)
