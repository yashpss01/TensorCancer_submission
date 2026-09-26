"""Out-of-fold group-aware matching on top of the pair model."""
import argparse
import difflib
import hashlib
import json
import math
import pathlib
import resource
import sys
import time

import numpy as np
from xgboost import XGBClassifier

ROOT=pathlib.Path(__file__).resolve().parents[3]
R5=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(R5))
import dataset
import train as first
from features import represent,dice,weighted
from improve import select
from common import load,dump

ART=ROOT/'artifacts/matching_round5'
EXTRA_NAMES=['pair_probability','group_best_probability','group_second_probability','group_other_best_probability','group_high_count','group_moderate_count','probability_rank_fraction','probability_gap_to_best','anchor_count','anchor_name_core_dice','anchor_name_char3_dice','anchor_name_key_dice','anchor_address_dice','anchor_address_idf_cosine','anchor_address_number_containment','anchor_name_address_joint','anchor_core_exact','anchor_address_exact','anchor_support_count','name_core_sequence','name_full_sequence','name_key_sequence','address_sequence','address_first_number_equal','address_long_number_conflict','address_query_numbers_contained','address_target_numbers_contained']


def digest(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def params():
    return dict(n_estimators=700,max_depth=7,learning_rate=.06,min_child_weight=5,subsample=.9,colsample_bytree=.9,reg_lambda=3,tree_method='hist',n_jobs=8,objective='binary:logistic',eval_metric='logloss',early_stopping_rounds=35,random_state=20260926)


def crossfit():
    if (ART/'train_oof_prob.npy').exists(): raise RuntimeError('OOF predictions exist')
    X,y,offsets,meta=first.arrays('train')
    queries=load(dataset.prior.FRESH/'queries.json')[:30000]
    assert len(queries)==meta['queries']
    folds=np.array([int.from_bytes(hashlib.sha256(q['entity_id'].encode()).digest()[:4],'big')%3 for q in queries],dtype=np.uint8)
    pair_fold=np.repeat(folds,np.diff(offsets))
    out=np.lib.format.open_memmap(ART/'train_oof_prob.npy',mode='w+',dtype='float32',shape=(len(y),))
    info=[]; start=time.monotonic()
    for fold in range(3):
        mask=pair_fold==fold
        model=XGBClassifier(**params())
        model.fit(X[~mask],y[~mask],eval_set=[(X[mask],y[mask])],verbose=False)
        out[mask]=model.predict_proba(X[mask])[:,1].astype('float32')
        out.flush()
        info.append(dict(fold=fold,groups=int((folds==fold).sum()),pairs=int(mask.sum()),best_iteration=int(model.best_iteration)))
        print('fold',fold,info[-1],round(time.monotonic()-start,1),flush=True)
    assert np.isfinite(out).all()
    dump(ART/'train_oof_meta.json',dict(folds=info,seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sha256=digest(ART/'train_oof_prob.npy'),train_meta_sha256=digest(ART/'train/meta.json')))


def extra(split):
    assert split in ('train','validation','holdout')
    out=ART/split/'group_features.npy'
    if out.exists(): raise RuntimeError('Group features exist')
    X,y,offsets,meta=first.arrays(split)
    prob_file=ART/('train_oof_prob.npy' if split=='train' else f'{split}_xgboost_prob.npy')
    prob=np.load(prob_file,mmap_mode='r')
    assert len(prob)==len(y)
    queries,raw,truth,expected=dataset.sources(split)
    records=dataset.load_records()
    _,address_weights,default=dataset.load_weights()
    A=np.lib.format.open_memmap(out,mode='w+',dtype='float32',shape=(len(y),len(EXTRA_NAMES)))
    start=time.monotonic()
    pos=0
    for gi,(q,r) in enumerate(zip(queries,dataset.read_jsonl(raw)),1):
        mids=select(r,dataset.THRESHOLD,dataset.CAP)
        lo,hi=int(offsets[gi-1]),int(offsets[gi])
        assert len(mids)==hi-lo and r['id']==q['entity_id']
        ps=np.asarray(prob[lo:hi])
        if not len(ps): continue
        order=np.argsort(-ps,kind='stable')
        top=order[:3]
        anchors=[int(i) for i in top if ps[i]>=.9]
        if not anchors: anchors=[int(order[0])]
        reps=[represent(*records[mid]) for mid in mids]
        qrep=represent(q['business_name'],q['business_address'],q['country'])
        best=float(ps[order[0]]);second=float(ps[order[1]]) if len(order)>1 else 0.
        high=int((ps>=.9).sum());moderate=int((ps>=.5).sum())
        ranks=np.empty(len(ps),dtype=np.int32);ranks[order]=np.arange(len(ps))
        for j,rep in enumerate(reps):
            other=[a for a in anchors if a!=j]
            values=[]
            for a in other:
                ar=reps[a]
                ad,_=weighted(rep['at'],ar['at'],address_weights,default)
                nc=dice(rep['nc'],ar['nc']);aa=dice(rep['at'],ar['at'])
                values.append((nc,dice(rep['cg'],ar['cg']),dice(rep['nb'],ar['nb']),aa,ad,len(rep['nums']&ar['nums'])/min(len(rep['nums']),len(ar['nums'])) if rep['nums'] and ar['nums'] else 0.,nc*aa,float(bool(rep['core']) and rep['core']==ar['core']),float(bool(rep['addr']) and rep['addr']==ar['addr']),float(ps[a])))
            maxima=[max(v[k] for v in values) if values else 0. for k in range(10)]
            support=sum(v[6]>.4 and v[9]>.9 for v in values)
            other_best=second if j==order[0] else best
            seq=lambda a,b:difflib.SequenceMatcher(None,a,b,autojunk=False).ratio() if a and b else 0.
            qfirst=next((x for x in qrep['addr'].split() if x.isdigit()),'')
            tfirst=next((x for x in rep['addr'].split() if x.isdigit()),'')
            qlong={x for x in qrep['nums'] if len(x)>=3}
            tlong={x for x in rep['nums'] if len(x)>=3}
            A[lo+j]=[float(ps[j]),best,second,other_best,high,moderate,float(ranks[j])/max(1,len(ps)-1),best-float(ps[j]),len(other),*maxima[:9],support,
                seq(qrep['core'],rep['core']),seq(qrep['name'],rep['name']),seq(qrep['nk'],rep['nk']),seq(qrep['addr'],rep['addr']),float(bool(qfirst and tfirst) and qfirst==tfirst),float(bool(qlong and tlong) and not (qlong&tlong)),float(bool(qrep['nums']) and qrep['nums']<=rep['nums']),float(bool(rep['nums']) and rep['nums']<=qrep['nums'])]
            pos+=1
        if gi%1000==0:
            A.flush()
            print('group',split,gi,round(time.monotonic()-start,1),flush=True)
    assert pos==len(y)
    A.flush()
    dump(ART/split/'group_features_meta.json',dict(split=split,features=EXTRA_NAMES,pairs=len(y),seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sha256=digest(out),first_stage_probability_sha256=digest(prob_file)))


def train():
    if (ART/'stage2_results.json').exists(): raise RuntimeError('Already trained')
    X,y,_,mt=first.arrays('train')
    VX,VY,offsets,mv=first.arrays('validation')
    A=np.load(ART/'train/group_features.npy',mmap_mode='r')
    VA=np.load(ART/'validation/group_features.npy',mmap_mode='r')
    assert len(A)==len(y) and len(VA)==len(VY)
    start=time.monotonic()
    train_x=np.concatenate((X,A),axis=1)
    val_x=np.concatenate((VX,VA),axis=1)
    model=XGBClassifier(n_estimators=900,max_depth=6,learning_rate=.05,min_child_weight=7,subsample=.9,colsample_bytree=.9,reg_lambda=3,tree_method='hist',n_jobs=8,objective='binary:logistic',eval_metric='logloss',early_stopping_rounds=40,random_state=20260927)
    model.fit(train_x,y,eval_set=[(val_x,VY)],verbose=False)
    path=ART/'xgboost_group.json';model.save_model(path)
    prob=model.predict_proba(val_x)[:,1].astype('float32')
    np.save(ART/'validation_group_prob.npy',prob)
    counts,masks=first.group_truth('validation')
    grid,best=first.optimize(prob,VY,offsets,counts,masks)
    dump(ART/'stage2_grid.json',grid)
    dump(ART/'stage2_results.json',dict(best=best,best_iteration=int(model.best_iteration),model_path=str(path.relative_to(ROOT)),model_sha256=digest(path),validation_prob_sha256=digest(ART/'validation_group_prob.npy'),extra_features=EXTRA_NAMES,seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print('Stage 2',best['threshold'],best['overall']['macro_f05'],best['countries']['India']['macro_f05'],round(time.monotonic()-start,1),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['crossfit','extra_train','extra_validation','extra_holdout','train']);a=p.parse_args()
    if a.stage.startswith('extra_'): extra(a.stage.removeprefix('extra_'))
    else: globals()[a.stage]()
