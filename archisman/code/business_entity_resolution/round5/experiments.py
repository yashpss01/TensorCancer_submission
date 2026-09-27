"""Development-only matcher comparisons; final holdout remains sealed."""
import argparse
import pathlib
import sys
import time

import numpy as np
from xgboost import XGBClassifier

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import train as first
from common import dump

ART=ROOT/'artifacts/matching_round5'


def run(name):
    if name not in ('pair_deep','group_without_pair_score','group_without_probabilities'):
        raise ValueError(name)
    out=ART/f'{name}_results.json'
    if out.exists(): raise RuntimeError('Experiment already exists')
    X,y,_,_=first.arrays('train')
    VX,VY,offsets,_=first.arrays('validation')
    A=np.load(ART/'train/group_features.npy',mmap_mode='r')
    VA=np.load(ART/'validation/group_features.npy',mmap_mode='r')
    if name=='pair_deep':
        tx,vx=X,VX
    elif name=='group_without_pair_score':
        tx=np.concatenate((X,A[:,1:]),axis=1)
        vx=np.concatenate((VX,VA[:,1:]),axis=1)
    else:
        tx=np.concatenate((X,A[:,8:]),axis=1)
        vx=np.concatenate((VX,VA[:,8:]),axis=1)
    model=XGBClassifier(n_estimators=1400,max_depth=9,learning_rate=.04,min_child_weight=2,subsample=.9,colsample_bytree=.9,reg_lambda=2,tree_method='hist',n_jobs=8,objective='binary:logistic',eval_metric='logloss',early_stopping_rounds=50,random_state=20260928)
    start=time.monotonic()
    model.fit(tx,y,eval_set=[(vx,VY)],verbose=False)
    path=ART/f'{name}.json';model.save_model(path)
    prob=model.predict_proba(vx)[:,1].astype('float32')
    np.save(ART/f'validation_{name}_prob.npy',prob)
    counts,masks=first.group_truth('validation')
    grid,best=first.optimize(prob,VY,offsets,counts,masks)
    dump(ART/f'{name}_grid.json',grid)
    dump(out,dict(best=best,best_iteration=int(model.best_iteration),seconds=time.monotonic()-start,model=str(path.relative_to(ROOT)),features=tx.shape[1]))
    print(name,best['threshold'],best['overall']['macro_f05'],best['countries']['India']['macro_f05'],round(time.monotonic()-start,1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name',choices=['pair_deep','group_without_pair_score','group_without_probabilities']);a=p.parse_args()
    run(a.name)
