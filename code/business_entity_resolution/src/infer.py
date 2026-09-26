"""Build a label-blind test index and emit both competition TSVs.

All inference dependencies are in src/ and ../models/. No training labels or
prior evaluation SQLite files are read by this entry point.
"""
import argparse
import hashlib
import itertools
import json
import pathlib
import resource
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
from xgboost import XGBClassifier

import blocking
import evaluate
from blocking import FTS_SQL,fields,rows,tokens
from features import FEATURE_NAMES,features,represent
from inference_features import EXTRA_NAMES,GROUP_NAMES,extra_vector,group_matrix
from inference_retrieval import Improved,select
from normalize import address,bigrams
from phonetic import grams

ROOT=pathlib.Path(__file__).resolve().parents[3]
DEFAULT_MODELS=pathlib.Path(__file__).resolve().parents[1]/'models'
ENGINE_BASE=None
ENGINE_EXTRA=None


def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify_models(model_dir):
    manifest=json.loads((model_dir/'manifest.json').read_text())
    for name,sha in manifest['sha256'].items():
        assert digest(model_dir/name)==sha,name
    assert manifest['selection']=={'threshold':0.74,'group_weight':0.4,'augmented_weight':0.6,'candidate_threshold':0.5,'candidate_cap':16}
    assert manifest['feature_names']==FEATURE_NAMES
    assert manifest['extra_feature_names']==EXTRA_NAMES
    assert manifest['group_feature_names']==GROUP_NAMES
    return manifest


def index(test_dir,index_dir):
    """Index every S2/S3 record; never inspect test labels or country categories."""
    targets=[test_dir/f'test_source{s}.tsv' for s in (2,3)]
    assert all(x.is_file() for x in targets),targets
    index_dir.mkdir(parents=True,exist_ok=True)
    db_path=index_dir/'index.sqlite';extra_path=index_dir/'extra.sqlite'
    if any((index_dir/x).exists() for x in ('index.sqlite','extra.sqlite','index_meta.json')):
        raise RuntimeError('Index artifacts already exist; use a new empty index directory')
    start=time.monotonic();n=0
    db=sqlite3.connect(db_path);extra=sqlite3.connect(extra_path)
    db.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-196608; CREATE TABLE records(id TEXT PRIMARY KEY,name TEXT,address TEXT,country TEXT);'+FTS_SQL)
    extra.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; CREATE VIRTUAL TABLE extra USING fts5(native,addr,n2,content="",detail=full);')
    rec_batch=[];search_batch=[];extra_batch=[]
    for source in targets:
        for record in rows(source):
            n+=1
            rec_batch.append((n,record['entity_id'],record['business_name'],record['business_address'],record['country']))
            search_batch.append((n,*fields(record),' '.join(grams(record['business_name']))))
            extra_batch.append((n,' '.join(tokens(record['business_name'])),' '.join(address(record['business_address'])),' '.join(sorted(bigrams(record['business_name'])))))
            if len(rec_batch)==10000:
                db.executemany('INSERT INTO records(rowid,id,name,address,country) VALUES(?,?,?,?,?)',rec_batch)
                db.executemany('INSERT INTO search(rowid,name,address,grams,phonetic) VALUES(?,?,?,?,?)',search_batch)
                extra.executemany('INSERT INTO extra(rowid,native,addr,n2) VALUES(?,?,?,?)',extra_batch)
                db.commit();extra.commit()
                rec_batch=[];search_batch=[];extra_batch=[]
                if n%100000==0:print('indexed',n,'targets in',round(time.monotonic()-start,1),'seconds',flush=True)
    if rec_batch:
        db.executemany('INSERT INTO records(rowid,id,name,address,country) VALUES(?,?,?,?,?)',rec_batch)
        db.executemany('INSERT INTO search(rowid,name,address,grams,phonetic) VALUES(?,?,?,?,?)',search_batch)
        extra.executemany('INSERT INTO extra(rowid,native,addr,n2) VALUES(?,?,?,?)',extra_batch)
    db.execute('CREATE VIRTUAL TABLE vocab USING fts5vocab(search,col)')
    extra.execute('CREATE VIRTUAL TABLE vocab USING fts5vocab(extra,col)')
    db.commit();extra.commit()
    assert db.execute('SELECT count(*) FROM records').fetchone()[0]==n
    db.close();extra.close()
    meta=dict(target_count=n,source_files={str(p):p.stat().st_size for p in targets},index_seconds=time.monotonic()-start,index_bytes=db_path.stat().st_size,extra_bytes=extra_path.stat().st_size,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (index_dir/'index_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Indexed all',n,'targets',flush=True)


def init_worker(index_dir):
    global ENGINE_BASE,ENGINE_EXTRA
    evaluate.ART=pathlib.Path(index_dir)
    ENGINE_BASE=evaluate.Retriever()
    ENGINE_EXTRA=Improved(index_dir)


def block(query):
    baseline=ENGINE_BASE.retrieve(query)
    improved=ENGINE_EXTRA.retrieve(query,baseline)
    assert baseline['id']==improved['id']==query['entity_id']
    return select(improved,.5,16)


def target_records(db,ids):
    out={}
    unique=sorted(set(ids))
    for start in range(0,len(unique),900):
        subset=unique[start:start+900]
        sql='SELECT id,name,address,country FROM records WHERE id IN ('+','.join('?'*len(subset))+')'
        out.update({mid:(name,addr,country) for mid,name,addr,country in db.execute(sql,subset)})
    assert len(out)==len(unique),'Candidate ID missing from target index'
    return out


def load_models(model_dir):
    verify_models(model_dir)
    first=XGBClassifier();first.load_model(model_dir/'xgboost.json')
    group=XGBClassifier();group.load_model(model_dir/'group_extra.json')
    augmented=XGBClassifier();augmented.load_model(model_dir/'augmented_0p15.json')
    token_weights=joblib.load(model_dir/'token_idf.joblib')
    char_weights=joblib.load(model_dir/'char_idf.joblib')
    return first,group,augmented,token_weights,char_weights


def match_batch(queries,candidate_lists,db,models):
    first,group,augmented,(nw,aw,default),(cn,ca,cdefault)=models
    assert len(queries)==len(candidate_lists)
    all_ids=list(itertools.chain.from_iterable(candidate_lists))
    records=target_records(db,all_ids)
    X=[];E=[];reps=[];offsets=[0]
    for query,mids in zip(queries,candidate_lists):
        qr=represent(query['business_name'],query['business_address'],query['country'])
        current=[]
        for mid in mids:
            tr=represent(*records[mid])
            X.append(features(qr,tr,nw,aw,default))
            E.append(extra_vector(qr,tr,cn,ca,cdefault))
            current.append(tr)
        reps.append(current);offsets.append(len(X))
    if not X:return [[] for _ in queries]
    X=np.asarray(X,dtype='float32');E=np.asarray(E,dtype='float32')
    assert X.shape[1]==len(FEATURE_NAMES) and E.shape[1]==len(EXTRA_NAMES)
    pair=np.concatenate((X,E),axis=1)
    first_prob=first.predict_proba(X)[:,1].astype('float32')
    A=np.concatenate([group_matrix(query,group_reps,first_prob[offsets[i]:offsets[i+1]],aw,default) for i,(query,group_reps) in enumerate(zip(queries,reps))],axis=0)
    assert A.shape==(len(X),len(GROUP_NAMES))
    group_input=np.concatenate((pair,A[:,1:]),axis=1)
    prob=(.4*group.predict_proba(group_input)[:,1]+.6*augmented.predict_proba(pair)[:,1]).astype('float32')
    assert np.isfinite(prob).all()
    return [[mid for mid,p in zip(mids,prob[offsets[i]:offsets[i+1]]) if p>=.74] for i,mids in enumerate(candidate_lists)]


def run(source1,index_dir,output_dir,model_dir,workers,batch_size,limit):
    if not source1.is_file():raise FileNotFoundError(source1)
    if not (index_dir/'index_meta.json').is_file():raise FileNotFoundError('Build the target index first')
    if limit is not None and output_dir.resolve()==(ROOT/'output').resolve():
        raise ValueError('A limited smoke run must use a separate output directory')
    models=load_models(model_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    matching=output_dir/'matching_results.tsv';candidate=output_dir/'candidate_pairs.tsv'
    temp_matching=output_dir/'matching_results.tsv.partial';temp_candidate=output_dir/'candidate_pairs.tsv.partial'
    if any(p.exists() for p in (matching,candidate,temp_matching,temp_candidate)):
        raise RuntimeError('Output already exists; choose a new empty output directory')
    db=sqlite3.connect(f'file:{index_dir/"index.sqlite"}?mode=ro',uri=True)
    seen=0;candidate_count=0;match_count=0;start=time.monotonic()
    source_rows=rows(source1)
    if limit is not None:source_rows=itertools.islice(source_rows,limit)
    with open(temp_matching,'x',encoding='utf-8') as mf,open(temp_candidate,'x',encoding='utf-8') as cf,ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(str(index_dir),)) as pool:
        mf.write('source1_entity_id\tmatched_entity_ids\n')
        cf.write('source1_entity_id\tcandidate_entity_ids\n')
        while True:
            queries=list(itertools.islice(source_rows,batch_size))
            if not queries:break
            if any(not all(k in q and q[k] is not None for k in ('entity_id','business_name','business_address','country')) for q in queries):
                raise ValueError('Malformed Source 1 row')
            candidate_lists=list(pool.map(block,queries,chunksize=8))
            matches=match_batch(queries,candidate_lists,db,models)
            for q,mids,chosen in zip(queries,candidate_lists,matches):
                assert set(chosen)<=set(mids)
                sid=q['entity_id']
                cf.write(sid+'\t'+','.join(mids)+'\n')
                mf.write(sid+'\t'+','.join(chosen)+'\n')
                seen+=1;candidate_count+=len(mids);match_count+=len(chosen)
            cf.flush();mf.flush()
            if seen%1000<batch_size:print('inferred',seen,'S1 rows in',round(time.monotonic()-start,1),'seconds',flush=True)
    db.close()
    temp_matching.rename(matching);temp_candidate.rename(candidate)
    meta=dict(source1_rows=seen,candidate_pairs=candidate_count,predicted_pairs=match_count,partial_run=limit is not None,seconds=time.monotonic()-start,matching_sha256=digest(matching),candidate_sha256=digest(candidate),model_manifest_sha256=digest(model_dir/'manifest.json'))
    (output_dir/'inference_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    i=sub.add_parser('index',help='Build a label-blind index of all test S2 and S3 records')
    i.add_argument('--test-dir',type=pathlib.Path,default=ROOT/'dataset/test')
    i.add_argument('--index-dir',type=pathlib.Path,default=ROOT/'inference_index/test')
    r=sub.add_parser('run',help='Write matching_results.tsv and candidate_pairs.tsv')
    r.add_argument('--source1',type=pathlib.Path,default=ROOT/'dataset/test/test_source1.tsv')
    r.add_argument('--index-dir',type=pathlib.Path,default=ROOT/'inference_index/test')
    r.add_argument('--output-dir',type=pathlib.Path,default=ROOT/'output')
    r.add_argument('--model-dir',type=pathlib.Path,default=DEFAULT_MODELS)
    r.add_argument('--workers',type=int,default=6)
    r.add_argument('--batch-size',type=int,default=256)
    r.add_argument('--limit',type=int,help='Smoke test only; creates incomplete output')
    a=p.parse_args()
    if a.command=='index':index(a.test_dir,a.index_dir)
    else:
        if a.workers<1 or a.batch_size<1 or a.limit is not None and a.limit<1:raise ValueError('workers, batch size, and limit must be positive')
        run(a.source1,a.index_dir,a.output_dir,a.model_dir,a.workers,a.batch_size,a.limit)


if __name__=='__main__':main()
