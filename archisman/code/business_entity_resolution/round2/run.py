import argparse,time,resource,hashlib,sqlite3,collections,os
os.environ["PYTHONHASHSEED"]="0"
from concurrent.futures import ProcessPoolExecutor
from common import *
from improve import Improved,select,baseline_set
import evaluate
ENGINE=None

def init(split,kind):
    global ENGINE
    if kind=='baseline':
        evaluate.ART=ART/'fresh'; ENGINE=evaluate.Retriever()
    else: ENGINE=Improved(split)
def work(item):
    q,base=item
    return ENGINE.retrieve(q) if base is None else ENGINE.retrieve(q,base)
def baselines(split):
    if split=='dev':
        out={}
        for part in ['dev','holdout']:
            for r in load(OLD/part/'rankings.json'): out[r['id']]=r
        return out
    return {r['id']:r for r in read_lines(ART/'fresh/baseline.jsonl')}
def read_lines(p):
    with open(p) as f:
        for line in f: yield json.loads(line)
def hashes():
    root=pathlib.Path(__file__).parent
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/n for n in ['common.py','normalize.py','improve.py','run.py','extra_index.py']]+[SRC/n for n in ['blocking.py','evaluate.py','phonetic.py']]}
def check_freeze():
    cfg=load(ART/'frozen_config.json')
    assert cfg['code_hashes']==hashes(),'Code changed after freeze'
    assert cfg['baseline_config_sha256']==hashlib.sha256((OLD/'frozen_config.json').read_bytes()).hexdigest()
    for rel,digest in cfg['fresh_input_hashes'].items():
        with open(ROOT/rel,'rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==digest, 'Frozen input changed'
    return cfg

def run(split,kind,limit=None):
    if split=='dev' and (ART/'FRESH_OPENED.json').exists(): raise RuntimeError('Fresh outcomes opened; development tuning is closed')
    if split=='fresh':
        check_freeze()
        if limit: raise RuntimeError('No pilot on fresh evaluation')
        marker=ART/'FRESH_OPENED.json'
        if kind=='baseline':
            if marker.exists(): raise RuntimeError('Fresh benchmark already opened')
            dump(marker,dict(opened_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),config_sha256=hashlib.sha256((ART/'frozen_config.json').read_bytes()).hexdigest()))
        elif not marker.exists(): raise RuntimeError('Evaluate baseline first')
    suffix=f'_pilot{limit}' if limit else ''
    path=ART/split/f'{kind}{suffix}.jsonl'
    if path.exists(): raise RuntimeError('Saved run exists; refusing overwrite')
    load_start=time.monotonic()
    queries=load(ART/split/'queries.json'); queries=queries[:limit] if limit else queries
    bases=None if kind=='baseline' else baselines(split)
    load_seconds=time.monotonic()-load_start
    start=time.monotonic(); count=0
    with open(path,'x') as out,ProcessPoolExecutor(max_workers=3,initializer=init,initargs=(split,kind)) as pool:
        for r in pool.map(work,((q,None if bases is None else bases[q['entity_id']]) for q in queries),chunksize=8):
            out.write(json.dumps(r,ensure_ascii=False)+'\n'); count+=1
            if count%100==0: out.flush(); print(split,kind,count,round(time.monotonic()-start,1),flush=True)
    dump(ART/split/f'{kind}{suffix}_runtime.json',dict(queries=count,seconds=time.monotonic()-start,parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,max_child_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,workers=3,load_seconds_excluded=True,input_load_seconds=load_seconds))

def metrics(rs,truth,n,threshold=None,cap=None,kind='improved'):
    chosen=[dict(id=r['id'],country=r['country'],ranked=(baseline_set(r) if kind=='baseline' else select(r,threshold,cap))) for r in rs]
    def m(q): return evaluate.measure(q,truth,10000,n)
    return dict(overall=m(chosen),countries={c:m([r for r in chosen if r['country']==c]) for c in sorted({r['country'] for r in chosen})})

def sweep(limit=None):
    if (ART/'FRESH_OPENED.json').exists(): raise RuntimeError('No tuning after fresh opening')
    suffix=f'_pilot{limit}' if limit else ''
    rs=list(read_lines(ART/'dev'/f'improved{suffix}.jsonl')); truth=load(ART/'dev/truth.json'); n=load(OLD/'index_meta.json')['target_count']; results=[]
    for cap in [4,8,12,16,24]:
        for threshold in [.5,.55,.6,.65,.7,.75,.8]:
            ms=metrics(rs,truth,n,threshold,cap)
            results.append(dict(cap=cap,threshold=threshold,**ms))
    dump(ART/'dev'/f'sweep{suffix}.json',results)
    feasible=[r for r in results if r['overall']['recall']>=.997 and r['countries']['India']['recall']>=.997]
    best=min(feasible,key=lambda r:r['overall']['candidates_mean']) if feasible else max(results,key=lambda r:(r['countries']['India']['recall'],-r['overall']['candidates_mean']))
    dump(ART/'dev'/f'proposed{suffix}.json',best)
    print(json.dumps(best,indent=2))

def freeze():
    if (ART/'frozen_config.json').exists(): raise RuntimeError('Already frozen')
    cfg=load(ART/'dev/proposed.json'); cfg.update(code_hashes=hashes(),baseline_config_sha256=hashlib.sha256((OLD/'frozen_config.json').read_bytes()).hexdigest(),frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),selection='Smallest development mean candidates satisfying 99.70% India and overall recall if feasible; otherwise maximum India recall within examined cap/cutoff grid')
    cfg['fresh_input_hashes']={}
    for name in ['queries.json','truth.json','index.sqlite','extra.sqlite','manifest.json']:
        path=ART/'fresh'/name
        with open(path,'rb') as f: cfg['fresh_input_hashes'][str(path.relative_to(ROOT))]=hashlib.file_digest(f,'sha256').hexdigest()
    dump(ART/'frozen_config.json',cfg)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['run','sweep','freeze']); p.add_argument('--split',default='dev',choices=['dev','fresh']); p.add_argument('--kind',default='improved',choices=['baseline','improved']); p.add_argument('--limit',type=int); a=p.parse_args()
    if a.command=='run': run(a.split,a.kind,a.limit)
    elif a.command=='sweep': sweep(a.limit)
    else: freeze()
