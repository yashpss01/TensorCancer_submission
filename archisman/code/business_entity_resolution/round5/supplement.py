"""Additional labeled positive pairs from disjoint training S1 groups."""
import hashlib
import itertools
import pathlib
import resource
import sys
import time

import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import dataset
import extra_pair
from features import FEATURE_NAMES,features,represent
from common import rows,dump,load

ART=ROOT/'artifacts/matching_round5/supplement'
START=80000
END=280000


def build():
    if ART.exists():raise RuntimeError('Supplement exists')
    ART.mkdir()
    start=time.monotonic()
    queries=list(itertools.islice(rows(ROOT/'dataset/train/train_source1.tsv'),START,END))
    assert len(queries)==END-START
    ids={q['entity_id'] for q in queries}
    assert len(ids)==len(queries)
    holdout={q['entity_id'] for q in load(dataset.prior.FRESH/'queries.json')[40000:60000]}
    assert not ids&holdout
    truth={}
    holdout_targets=set()
    for r in rows(ROOT/'dataset/train/train_ground_truth.tsv'):
        if r['source1_entity_id'] in ids:
            truth[r['source1_entity_id']]=r['matched_entity_ids'].split(',') if r['matched_entity_ids'] else []
        elif r['source1_entity_id'] in holdout and r['matched_entity_ids']:
            holdout_targets.update(r['matched_entity_ids'].split(','))
    assert len(truth)==len(queries)
    required={mid for mids in truth.values() for mid in mids}
    assert sum(map(len,truth.values()))==len(required), 'One target belongs to multiple supplemental groups'
    assert not required&holdout_targets, 'Supplement overlaps holdout target labels'
    records={}
    for source in [2,3]:
        for r in rows(ROOT/f'dataset/train/train_source{source}.tsv'):
            if r['entity_id'] in required:
                records[r['entity_id']]=(r['business_name'],r['business_address'],r['country'])
    assert len(records)==len(required)
    nw,aw,default=dataset.load_weights()
    cn,ca,cdefault=extra_pair.idf()
    X=np.lib.format.open_memmap(ART/'X.npy',mode='w+',dtype='float32',shape=(len(required),len(FEATURE_NAMES)))
    E=np.lib.format.open_memmap(ART/'E.npy',mode='w+',dtype='float32',shape=(len(required),len(extra_pair.NAMES)))
    pos=0
    for i,q in enumerate(queries,1):
        qr=represent(q['business_name'],q['business_address'],q['country'])
        for mid in truth[q['entity_id']]:
            tr=represent(*records[mid])
            X[pos]=features(qr,tr,nw,aw,default)
            E[pos]=extra_pair.vector(qr,tr,cn,ca,cdefault)
            pos+=1
        if i%10000==0:
            X.flush();E.flush()
            print('supplement',i,pos,round(time.monotonic()-start,1),flush=True)
    assert pos==len(required)
    X.flush();E.flush()
    dump(ART/'meta.json',dict(s1_original_rows=f'{START+1}–{END}',groups=len(queries),positive_pairs=pos,all_target_records_present=True,no_shared_labeled_target_with_holdout=True,seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,features=FEATURE_NAMES,extra_features=extra_pair.NAMES))


if __name__=='__main__':build()
