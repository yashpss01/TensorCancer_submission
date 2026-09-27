"""Additional typo, domain, character-IDF, and numeric pair features."""
import argparse
from collections import Counter
import difflib
import math
import pathlib
import re
import resource
import sys
import time

import joblib
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import dataset
from features import represent,weighted
from improve import select
from common import load,dump

ART=ROOT/'artifacts/matching_round5'
NAMES=['core_sequence_ratio','full_name_sequence_ratio','address_sequence_ratio','core_sorted_sequence_ratio','name_key_sequence_ratio','core_longest_match','address_longest_match','core_soft_overlap','core_soft_containment','core_char_idf_cosine','address_char_idf_cosine','target_address_semantically_missing','target_domain_name','query_domain_name','first_address_number_equal','long_address_number_conflict','address_number_query_contained','address_number_target_contained','name_token_count_difference','address_token_count_difference']


def idf():
    path=ART/'char_idf.joblib'
    if path.exists():return joblib.load(path)
    records=dataset.load_records()
    names=Counter();addresses=Counter()
    for name,addr,country in records.values():
        rep=represent(name,addr,country)
        names.update(rep['cg']);addresses.update(rep['ag'])
    n=len(records)
    result=({x:math.log1p(n/(1+c)) for x,c in names.items()},{x:math.log1p(n/(1+c)) for x,c in addresses.items()},math.log1p(n))
    joblib.dump(result,path)
    return result


def seq(a,b):
    if not a or not b:return 0.,0.
    matcher=difflib.SequenceMatcher(None,a,b,autojunk=False)
    return matcher.ratio(),matcher.find_longest_match(0,len(a),0,len(b)).size/min(len(a),len(b))


def soft(a,b):
    if not a or not b:return 0.,0.
    x=a.split();y=b.split()
    total=0.
    for word in x:
        best=max((difflib.SequenceMatcher(None,word,other,autojunk=False).ratio() for other in y),default=0.)
        if best>=.55:total+=best
    return total/max(len(x),len(y)),total/min(len(x),len(y))


def vector(q,t,nw,aw,default):
    core,long_core=seq(q['core'],t['core'])
    name,_=seq(q['name'],t['name'])
    addr,long_addr=seq(q['addr'],t['addr'])
    sorted_core,_=seq(' '.join(sorted(q['core'].split())),' '.join(sorted(t['core'].split())))
    key,_=seq(q['nk'],t['nk'])
    ss,sc=soft(q['core'],t['core'])
    nc,_=weighted(q['cg'],t['cg'],nw,default)
    ac,_=weighted(q['ag'],t['ag'],aw,default)
    qfirst=next((x for x in q['addr'].split() if x.isdigit()),'')
    tfirst=next((x for x in t['addr'].split() if x.isdigit()),'')
    qlong={x for x in q['nums'] if len(x)>=3}
    tlong={x for x in t['nums'] if len(x)>=3}
    missing=not t['addr'] or t['addr'] in {'n a','na','none','unknown','not available','nil'}
    return [core,name,addr,sorted_core,key,long_core,long_addr,ss,sc,nc,ac,float(missing),float('com' in t['nt'] or 'www' in t['nt']),float('com' in q['nt'] or 'www' in q['nt']),float(bool(qfirst and tfirst) and qfirst==tfirst),float(bool(qlong and tlong) and not (qlong&tlong)),float(bool(q['nums']) and q['nums']<=t['nums']),float(bool(t['nums']) and t['nums']<=q['nums']),abs(len(q['nt'])-len(t['nt'])),abs(len(q['at'])-len(t['at']))]


def build(split):
    assert split in ('train','validation','holdout')
    out=ART/split/'extra_pair.npy'
    if out.exists():raise RuntimeError('Extra pair features exist')
    queries,raw,truth,expected=dataset.sources(split)
    offsets=np.load(ART/split/'offsets.npy')
    records=dataset.load_records()
    nw,aw,default=idf()
    X=np.lib.format.open_memmap(out,mode='w+',dtype='float32',shape=(int(offsets[-1]),len(NAMES)))
    start=time.monotonic();index=0
    for gi,(q,r) in enumerate(zip(queries,dataset.read_jsonl(raw)),1):
        mids=select(r,.5,16)
        assert r['id']==q['entity_id'] and len(mids)==offsets[gi]-offsets[gi-1]
        qr=represent(q['business_name'],q['business_address'],q['country'])
        for mid in mids:
            X[index]=vector(qr,represent(*records[mid]),nw,aw,default)
            index+=1
        if gi%1000==0:
            X.flush();print(split,gi,round(time.monotonic()-start,1),flush=True)
    assert index==offsets[-1]
    X.flush()
    dump(ART/split/'extra_pair_meta.json',dict(split=split,pairs=index,features=NAMES,seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['train','validation','holdout']);a=p.parse_args()
    build(a.split)
