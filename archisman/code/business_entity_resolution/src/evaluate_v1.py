"""Reproducible development sweeps and sealed, once-only holdout evaluation."""
import argparse, collections, csv, hashlib, json, math, pathlib, resource, sqlite3, statistics, time
from blocking import ART, ROOT, dump, fields, tokens

def quote(t): return '"'+t.replace('"','""')+'"'

class Retriever:
    def __init__(self):
        self.db=sqlite3.connect(f'file:{ART}/index.sqlite?mode=ro',uri=True)
        self.db.execute('PRAGMA cache_size=-131072')
        self.n=json.loads((ART/'index_meta.json').read_text())['target_count']; self.cache={}
    def df(self,col,t):
        key=(col,t)
        if key not in self.cache:
            self.cache[key]=next((r[0] for r in self.db.execute('SELECT doc FROM vocab WHERE term=? AND col=?',(t,col))),0)
        return self.cache[key]
    def rare(self,col,ts,k):
        return sorted(set(ts), key=lambda t:(self.df(col,t) or self.n,t))[:k]
    def search(self,expr,k):
        if not expr: return []
        return [r[0] for r in self.db.execute('SELECT rowid FROM search WHERE search MATCH ? ORDER BY rank LIMIT ?',(expr,k))]
    def retrieve(self,q):
        ns,ads,gs=fields(q); n=ns.split(); a=ads.split(); g=gs.split()
        rn=self.rare('name',n,4); ra=self.rare('address',a,4); rg=self.rare('grams',g,5)
        expr=lambda col,ts:' OR '.join(col+':'+quote(t) for t in ts)
        routes={}
        routes['name']=self.search(expr('name',rn),40)
        routes['address']=self.search(expr('address',ra),40)
        # Joint field evidence rescues popular names and partial addresses.
        if rn and ra:
            routes['joint']=self.search('('+expr('name',rn)+') AND ('+expr('address',ra)+')',40)
        # Require two rare trigrams, avoiding a global single-character/gram expansion.
        pairs=['(grams:'+quote(x)+' AND grams:'+quote(y)+')' for i,x in enumerate(rg) for y in rg[i+1:]]
        routes['char']=self.search(' OR '.join(pairs),40) if pairs else []
        ids=sorted(set().union(*map(set,routes.values())))
        records={r[0]:r[1:] for r in self.db.execute('SELECT rowid,id,name,address,country FROM records WHERE rowid IN ('+','.join('?'*len(ids))+')',ids)} if ids else {}
        def weighted(col,x,y):
            x,y=set(x),set(y)
            if not x or not y: return 0.
            w=lambda t:math.log(1+self.n/(1+self.df(col,t)))
            return sum(w(t) for t in x&y)/sum(w(t) for t in x|y)
        def dice(x,y):
            x,y=set(x),set(y)
            return 2*len(x&y)/(len(x)+len(y)) if x and y else 0.
        scored=[]
        for rid,(eid,name,addr,country) in records.items():
            nn,aa,gg=fields(dict(business_name=name,business_address=addr))
            sn=max(weighted('name',n,nn.split()),dice(g,gg.split()))
            sa=weighted('address',a,aa.split())
            score=max(.55*sn+.45*sa,.78*sa,.78*sn)+.1*min(sn,sa)
            scored.append((score,eid))
        scored.sort(key=lambda z:(-z[0],z[1]))
        return dict(id=q['entity_id'],country=q['country'],ranked=[x[1] for x in scored],scores=[x[0] for x in scored],routes={k:[records[i][0] for i in v] for k,v in routes.items()},raw_count=len(ids))

def choose(q,k,threshold=0.,minimum=0):
    return [m for i,m in enumerate(q['ranked'][:k]) if i<minimum or q['scores'][i]>=threshold] if 'scores' in q else q['ranked'][:k]

def measure(queries,truth,k,target_count,route=None,threshold=0.,minimum=0):
    tp=fn=fp=0; sizes=[]; oracle=[]; affected=0
    for q in queries:
        true=set(truth[q['id']]); cand=set(q['routes'][route] if route else choose(q,k,threshold,minimum)); hit=len(true&cand)
        tp+=hit; fn+=len(true)-hit; fp+=len(cand-true); sizes.append(len(cand)); affected+=bool(true-cand)
        oracle.append(1. if not true else (1.25*hit/(hit+.25*len(true)) if hit else 0.))
    universe=len(queries)*target_count; tn=universe-tp-fn-fp
    recall=tp/(tp+fn) if tp+fn else 1.; precision=tp/(tp+fp) if tp+fp else 0.
    return dict(queries=len(queries),targets=target_count,comparison_space=universe,TP=tp,FN=fn,FP=fp,TN=tn,recall=recall,fn_rate=1-recall,precision=precision,reduction_ratio=1-(tp+fp)/universe,specificity=tn/(tn+fp),candidate_f1=2*tp/(2*tp+fp+fn),candidates=tp+fp,candidates_mean=statistics.mean(sizes),candidates_p95=sorted(sizes)[math.ceil(.95*len(sizes))-1],candidates_max=max(sizes),oracle_macro_f05=statistics.mean(oracle),entities_with_misses=affected,singletons=sum(not truth[q['id']] for q in queries))

def run(split):
    out=ART/split; out.mkdir(exist_ok=True)
    if (out/'rankings.json').exists(): raise RuntimeError('Existing rankings: use summarize for dev; no repeated holdout retrieval')
    if split=='holdout':
        if not (ART/'frozen_config.json').exists(): raise RuntimeError('Freeze dev choice before holdout')
        marker=ART/'HOLDOUT_OPENED.json'
        if marker.exists(): raise RuntimeError('Holdout already opened; refusing reuse')
        dump(marker,dict(opened=time.time(),config_sha256=hashlib.sha256((ART/'frozen_config.json').read_bytes()).hexdigest()))
    start=time.monotonic(); retriever=Retriever(); result=[]
    queries=[q for q in json.loads((ART/'queries.json').read_text()) if q['split']==split]
    for i,q in enumerate(queries):
        result.append(retriever.retrieve(q))
        if (i+1)%25==0: print(f'{split} {i+1}/{len(queries)} {time.monotonic()-start:.1f}s',flush=True)
    dump(out/'rankings.json',result)
    dump(out/'runtime.json',dict(seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,raw_pairs=sum(q['raw_count'] for q in result)))
    summarize(split)

def summarize(split):
    queries=json.loads((ART/split/'rankings.json').read_text()); truth=json.loads((ART/f'{split}_truth.json').read_text()); n=json.loads((ART/'index_meta.json').read_text())['target_count']
    if split=='dev':
        sweep={str(k):measure(queries,truth,k,n) for k in [5,10,15,20,30,40,60,80,120,160]}
        sweep.update({r:measure(queries,truth,0,n,r) for r in ['name','address','joint','char']})
        dump(ART/split/'sweep.json',sweep)
        print(json.dumps({k:{x:v[x] for x in ['recall','candidates_mean','FN']} for k,v in sweep.items()},indent=2))
    else:
        cfg=json.loads((ART/'frozen_config.json').read_text()); export(split,cfg['k'],queries,truth,n,cfg.get('threshold',0.),cfg.get('minimum',0))

def export(split,k,queries=None,truth=None,n=None,threshold=0.,minimum=0):
    queries=queries or json.loads((ART/split/'rankings.json').read_text()); truth=truth or json.loads((ART/f'{split}_truth.json').read_text()); n=n or json.loads((ART/'index_meta.json').read_text())['target_count']
    metrics={'overall':measure(queries,truth,k,n,threshold=threshold,minimum=minimum),'countries':{c:measure([q for q in queries if q['country']==c],truth,k,n,threshold=threshold,minimum=minimum) for c in sorted({q['country'] for q in queries})}}
    dump(ART/split/'metrics.json',metrics)
    with open(ART/split/'candidate_pairs.tsv','w') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for q in queries: f.write(q['id']+'\t'+','.join(choose(q,k,threshold,minimum))+'\n')
    raw={q['entity_id']:q for q in json.loads((ART/'queries.json').read_text())}; db=sqlite3.connect(ART/'index.sqlite'); misses=[]
    for q in queries:
        for m in sorted(set(truth[q['id']])-set(choose(q,k,threshold,minimum))):
            target=db.execute('SELECT id,name,address,country FROM records WHERE id=?',(m,)).fetchone()
            assert target, 'Truth ID absent from full target index'
            misses.append(dict(query=raw[q['id']],target=target,retrieved_rank=q['ranked'].index(m)+1 if m in q['ranked'] else None))
    dump(ART/split/'misses.json',misses)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['run','summarize','export']); p.add_argument('split',choices=['dev','holdout']); p.add_argument('--k',type=int,default=20); a=p.parse_args()
    if a.action=='export':
        if a.split!='dev': raise RuntimeError('Holdout exports only via frozen run')
        export(a.split,a.k)
    else: globals()[a.action](a.split)
