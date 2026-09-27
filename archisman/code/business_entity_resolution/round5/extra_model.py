"""Development comparison with richer string features."""
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


def run(kind):
    if kind not in ('pair_extra','group_extra'):raise ValueError(kind)
    result=ART/f'{kind}_results.json'
    if result.exists():raise RuntimeError('Model exists')
    X,y,_,_=first.arrays('train')
    VX,VY,offsets,_=first.arrays('validation')
    E=np.load(ART/'train/extra_pair.npy',mmap_mode='r')
    VE=np.load(ART/'validation/extra_pair.npy',mmap_mode='r')
    parts=[X,E];vparts=[VX,VE]
    if kind=='group_extra':
        A=np.load(ART/'train/group_features.npy',mmap_mode='r')
        VA=np.load(ART/'validation/group_features.npy',mmap_mode='r')
        parts.append(A[:,1:]);vparts.append(VA[:,1:])
    tx=np.concatenate(parts,axis=1);vx=np.concatenate(vparts,axis=1)
    model=XGBClassifier(n_estimators=1300,max_depth=8,learning_rate=.05,min_child_weight=3,subsample=.9,colsample_bytree=.9,reg_lambda=2,tree_method='hist',n_jobs=8,objective='binary:logistic',eval_metric='logloss',early_stopping_rounds=50,random_state=20260930)
    start=time.monotonic()
    model.fit(tx,y,eval_set=[(vx,VY)],verbose=False)
    path=ART/f'{kind}.json';model.save_model(path)
    p=model.predict_proba(vx)[:,1].astype('float32')
    np.save(ART/f'validation_{kind}_prob.npy',p)
    counts,masks=first.group_truth('validation')
    grid,best=first.optimize(p,VY,offsets,counts,masks)
    dump(ART/f'{kind}_grid.json',grid)
    dump(result,dict(best=best,best_iteration=int(model.best_iteration),seconds=time.monotonic()-start,features=tx.shape[1],model=str(path.relative_to(ROOT))))
    print(kind,best['threshold'],best['overall']['macro_f05'],best['countries']['India']['macro_f05'],round(time.monotonic()-start,1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['pair_extra','group_extra']);a=p.parse_args()
    run(a.kind)
