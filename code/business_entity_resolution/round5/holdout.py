"""Freeze a matcher, then evaluate once on untouched S1 rows 60,001–80,000."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import importlib
import itertools
import json
import pathlib
import resource
import sys
import time

import numpy as np
from xgboost import XGBClassifier

ROOT=pathlib.Path(__file__).resolve().parents[3]
R5=pathlib.Path(__file__).resolve().parent
R2=ROOT/'code/business_entity_resolution/round2'
R3=ROOT/'code/business_entity_resolution/round3'
sys.path[:0]=[str(R5),str(R3),str(R2)]
import dataset
import train
from common import load,dump

ART=ROOT/'artifacts/matching_round5'
OUT=ART/'holdout'
N=20000
BLEND_GROUP=.4
BLEND_AUGMENTED=.6
THRESHOLD=.74
ENGINE=None


def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def frozen():
    cfg=load(OUT/'frozen_model.json')
    assert cfg['holdout_rows']=='60001–80000'
    assert cfg['threshold']==THRESHOLD and cfg['blend_group']==BLEND_GROUP and cfg['blend_augmented']==BLEND_AUGMENTED
    for group in ['code_hashes','model_hashes','input_hashes']:
        for rel,sha in cfg[group].items():assert digest(ROOT/rel)==sha,rel
    return cfg


def freeze():
    if OUT.exists():raise RuntimeError('Holdout directory already exists')
    dataset.prior.verify_inputs()
    p=np.load(ART/'validation_group_extra_prob.npy')
    q=np.load(ART/'validation_augmented_0p15_prob.npy')
    _,y,offsets,_=train.arrays('validation')
    counts,masks=train.group_truth('validation')
    development=train.score_prob(BLEND_GROUP*p+BLEND_AUGMENTED*q,y,offsets,counts,masks,THRESHOLD)
    code=[R5/x for x in ['features.py','extra_pair.py','stage2.py','dataset.py','train.py','augmented_model.py','holdout.py']]
    code += [R2/x for x in ['common.py','normalize.py','improve.py','run.py','extra_index.py']]
    models=[ART/x for x in ['xgboost.json','group_extra.json','augmented_0p15.json']]
    inputs=[ART/x for x in ['train/meta.json','validation/meta.json','supplement/meta.json','group_extra_results.json','augmented_0p15_results.json']]
    OUT.mkdir()
    cfg=dict(holdout_rows='60001–80000',target_pool='Sealed reduced 409141-target pool',threshold=THRESHOLD,blend_group=BLEND_GROUP,blend_augmented=BLEND_AUGMENTED,development=development,code_hashes={str(x.relative_to(ROOT)):digest(x) for x in code},model_hashes={str(x.relative_to(ROOT)):digest(x) for x in models},input_hashes={str(x.relative_to(ROOT)):digest(x) for x in inputs},frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),selection='Two development models blended at fixed 0.4/0.6 weights; all choices fixed before holdout retrieval or scoring')
    dump(OUT/'frozen_model.json',cfg)
    print('Frozen matcher. Development macro F0.5:',development['overall']['macro_f05'],flush=True)


def init_worker(kind):
    global ENGINE
    if kind=='baseline':
        ev=importlib.import_module('evaluate')
        ev.ART=dataset.prior.FRESH
        ENGINE=ev.Retriever()
    else:
        imp=importlib.import_module('improve')
        imp.ART=dataset.prior.ART
        ENGINE=imp.Improved('fresh')


def retrieve(item):
    q,base=item
    return ENGINE.retrieve(q) if base is None else ENGINE.retrieve(q,base)


def run(kind):
    frozen()
    assert kind in ('baseline','improved')
    path=OUT/f'{kind}.jsonl'
    if path.exists():raise RuntimeError('Retrieval already exists')
    if kind=='improved':assert (OUT/'baseline_runtime.json').exists()
    queries=load(dataset.prior.FRESH/'queries.json')[40000:60000]
    assert len(queries)==N and len({q['entity_id'] for q in queries})==N
    bases=dataset.read_jsonl(OUT/'baseline.jsonl') if kind=='improved' else itertools.repeat(None)
    start=time.monotonic();count=0
    with open(path,'x') as f,ProcessPoolExecutor(max_workers=6,initializer=init_worker,initargs=(kind,)) as pool:
        items=zip(queries,bases)
        while count<N:
            batch=list(itertools.islice(items,1000))
            assert batch
            for result in pool.map(retrieve,batch,chunksize=8):
                f.write(json.dumps(result,ensure_ascii=False)+'\n');count+=1
            f.flush()
            print(kind,count,round(time.monotonic()-start,1),flush=True)
    assert count==N
    dump(OUT/f'{kind}_runtime.json',dict(queries=count,seconds=time.monotonic()-start,workers=6,parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,max_child_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,sha256=digest(path)))


def first_prob():
    frozen()
    path=ART/'holdout_xgboost_prob.npy'
    if path.exists():raise RuntimeError('First-stage scores exist')
    X,y,_,_=train.arrays('holdout')
    model=XGBClassifier();model.load_model(ART/'xgboost.json')
    p=model.predict_proba(X)[:,1].astype('float32')
    assert len(p)==len(y) and np.isfinite(p).all()
    np.save(path,p)
    print('First-stage probability pairs:',len(p),flush=True)


def predict():
    frozen()
    path=ART/'holdout_blend_prob.npy'
    if path.exists():raise RuntimeError('Predictions exist')
    X,y,_,_=train.arrays('holdout')
    E=np.load(OUT/'extra_pair.npy',mmap_mode='r')
    A=np.load(OUT/'group_features.npy',mmap_mode='r')
    assert len(X)==len(E)==len(A)==len(y)
    assert A.shape[1]>1
    base=np.concatenate((X,E),axis=1)
    group=np.concatenate((base,A[:,1:]),axis=1)
    gm=XGBClassifier();gm.load_model(ART/'group_extra.json')
    am=XGBClassifier();am.load_model(ART/'augmented_0p15.json')
    p=(BLEND_GROUP*gm.predict_proba(group)[:,1]+BLEND_AUGMENTED*am.predict_proba(base)[:,1]).astype('float32')
    assert len(p)==len(y) and np.isfinite(p).all()
    np.save(path,p)
    dump(OUT/'predict_meta.json',dict(pairs=len(p),probability_sha256=digest(path),group_feature_sha256=digest(OUT/'group_features.npy'),pair_feature_sha256=digest(OUT/'extra_pair.npy')))
    print('Predicted',len(p),'pairs',flush=True)


def score():
    cfg=frozen()
    if (OUT/'results.json').exists():raise RuntimeError('Holdout already scored')
    p=np.load(ART/'holdout_blend_prob.npy',mmap_mode='r')
    X,y,offsets,meta=train.arrays('holdout')
    assert len(p)==len(y)==meta['pairs']
    counts,masks=train.group_truth('holdout')
    result=train.score_prob(p,y,offsets,counts,masks,THRESHOLD)
    oracle=train.score_prob(y.astype('float32'),y,offsets,counts,masks,.5)
    def blocker(mask):
        starts=offsets[:-1][mask];ends=offsets[1:][mask]
        hits=int(sum(int(y[a:b].sum()) for a,b in zip(starts,ends)))
        truths=int(counts[mask].sum())
        return dict(queries=int(mask.sum()),candidate_pairs=int(sum(ends-starts)),true_links=truths,retrieved_true_links=hits,recall=hits/truths if truths else 1.)
    bm=dict(overall=blocker(np.ones(len(counts),dtype=bool)),countries={c:blocker(m) for c,m in masks.items()})
    output=dict(status='PASS',frozen_config_sha256=digest(OUT/'frozen_model.json'),holdout_rows=cfg['holdout_rows'],target_pool=cfg['target_pool'],selection_threshold=THRESHOLD,development=cfg['development'],holdout=result,blocking=bm,oracle_macro_f05=dict(overall=oracle['overall']['macro_f05'],countries={c:v['macro_f05'] for c,v in oracle['countries'].items()}),probability_sha256=digest(ART/'holdout_blend_prob.npy'),feature_meta_sha256=digest(OUT/'meta.json'))
    dump(OUT/'results.json',output)
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','baseline','improved','first_prob','predict','score']);a=p.parse_args()
    run(a.stage) if a.stage in ('baseline','improved') else globals()[a.stage]()
