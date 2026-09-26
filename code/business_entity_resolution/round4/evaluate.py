"""Fresh, paired validation of a selection change using the sealed round-3 pool."""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import importlib
import itertools
import json
import pathlib
import resource
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
R2 = ROOT / 'code/business_entity_resolution/round2'
R3 = ROOT / 'code/business_entity_resolution/round3'
sys.path[:0] = [str(R3), str(R2)]
import evaluate_60k as prior
from common import load, dump
from improve import baseline_set, select

ART = ROOT / 'artifacts/blocking_round4'
FRESH = ART / 'fresh'
SOURCE_POOL = prior.FRESH
N = 10000
OLD_THRESHOLD, OLD_CAP = .7, 4
NEW_THRESHOLD, NEW_CAP = .5, 16
ENGINE = None


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify_prior():
    prior.verify_inputs()
    assert load(prior.FRESH/'validation.json')['status'] == 'PASS'


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def development():
    verify_prior()
    if ART.exists():
        raise RuntimeError('Round-4 artifacts already exist')
    ART.mkdir()
    q = load(SOURCE_POOL/'queries.json')[:30000]
    truth = load(SOURCE_POOL/'truth.json')
    options = [(th, cap) for th in [.7,.65,.6,.55,.5,.45] for cap in [4,8,12,16,24]]
    sums = {o:dict(tp=0, india_tp=0, candidates=0) for o in options}
    india_total = sum(len(truth[x['entity_id']]) for x in q if x['country']=='India')
    total = sum(len(truth[x['entity_id']]) for x in q)
    for x,r in zip(q,read_jsonl(SOURCE_POOL/'improved.jsonl')):
        true = set(truth[x['entity_id']])
        for option in options:
            chosen = set(select(r,*option))
            hit = len(true&chosen)
            sums[option]['tp'] += hit
            sums[option]['india_tp'] += hit if x['country']=='India' else 0
            sums[option]['candidates'] += len(chosen)
    grid = [dict(threshold=th,cap=cap,overall_recall=sums[th,cap]['tp']/total,india_recall=sums[th,cap]['india_tp']/india_total,mean_candidates=sums[th,cap]['candidates']/30000) for th,cap in options]
    dump(ART/'development_grid.json',grid)
    choice = next(x for x in grid if x['threshold']==NEW_THRESHOLD and x['cap']==NEW_CAP)
    assert choice['india_recall'] >= .9955
    dump(ART/'development_choice.json',dict(**choice,selection_reason='At least 99.55% India recall on inspected development slice, with fewer than 50 candidates per S1 on average; preserve all original candidates'))
    print(json.dumps(choice,indent=2))


def prepare():
    verify_prior()
    assert (ART/'development_choice.json').exists()
    if FRESH.exists():
        raise RuntimeError('Fresh split exists')
    FRESH.mkdir()
    all_q = load(SOURCE_POOL/'queries.json')
    queries = all_q[30000:40000]
    assert len(queries)==N and len({q['entity_id'] for q in queries})==N
    assert not ({q['entity_id'] for q in all_q[:30000]} & {q['entity_id'] for q in queries})
    all_truth = load(SOURCE_POOL/'truth.json')
    truth = {q['entity_id']:all_truth[q['entity_id']] for q in queries}
    dump(FRESH/'queries.json',queries)
    dump(FRESH/'truth.json',truth)
    dump(FRESH/'manifest.json',dict(rows='50001–60000 inclusive in original S1 order',reference_count=N,target_pool=str(SOURCE_POOL.relative_to(ROOT)),target_count=load(SOURCE_POOL/'index_meta.json')['target_count'],target_policy='Same sealed 409141-target pool built in round 3; all this batch positives present',development='Rows 20001–50000 only; no fresh outcome tuning',original_threshold=OLD_THRESHOLD,original_cap=OLD_CAP,new_threshold=NEW_THRESHOLD,new_cap=NEW_CAP))
    print('Prepared next 10k fresh references')


def freeze():
    verify_prior()
    if (ART/'frozen_config.json').exists():
        raise RuntimeError('Already frozen')
    assert (FRESH/'manifest.json').exists()
    paths = [R2/x for x in ['common.py','normalize.py','improve.py','run.py','extra_index.py']]
    paths += [prior.SRC/x for x in ['blocking.py','evaluate.py','phonetic.py']]
    paths += [pathlib.Path(__file__)]
    cfg = dict(threshold=NEW_THRESHOLD,cap=NEW_CAP,source_pool_seal=digest(prior.ART/'sealed.json'),original_round2_config=digest(prior.R2_ART/'frozen_config.json'),code_hashes={str(p.relative_to(ROOT)):digest(p) for p in paths},input_hashes={str((FRESH/x).relative_to(ROOT)):digest(FRESH/x) for x in ['queries.json','truth.json','manifest.json']},pool_hashes={str((SOURCE_POOL/x).relative_to(ROOT)):digest(SOURCE_POOL/x) for x in ['index.sqlite','extra.sqlite','index_meta.json']},development_choice_sha256=digest(ART/'development_choice.json'),frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    dump(ART/'frozen_config.json',cfg)
    print('Frozen validation configuration')


def verify_freeze():
    verify_prior()
    cfg=load(ART/'frozen_config.json')
    assert cfg['source_pool_seal']==digest(prior.ART/'sealed.json')
    assert cfg['original_round2_config']==digest(prior.R2_ART/'frozen_config.json')
    assert cfg['development_choice_sha256']==digest(ART/'development_choice.json')
    for group in ['code_hashes','input_hashes','pool_hashes']:
        for rel,sha in cfg[group].items():
            assert digest(ROOT/rel)==sha,rel
    assert cfg['threshold']==NEW_THRESHOLD and cfg['cap']==NEW_CAP
    return cfg


def init_worker(kind):
    global ENGINE
    if kind=='baseline':
        module=importlib.import_module('evaluate')
        module.ART=SOURCE_POOL
        ENGINE=module.Retriever()
    else:
        module=importlib.import_module('improve')
        module.ART=prior.ART
        ENGINE=module.Improved('fresh')


def work(item):
    q,base=item
    return ENGINE.retrieve(q) if base is None else ENGINE.retrieve(q,base)


def run(kind):
    verify_freeze()
    assert kind in ('baseline','improved')
    path=FRESH/f'{kind}.jsonl'
    if path.exists(): raise RuntimeError('Output exists')
    if kind=='improved': assert (FRESH/'baseline_runtime.json').exists()
    queries=load(FRESH/'queries.json')
    bases=read_jsonl(FRESH/'baseline.jsonl') if kind=='improved' else itertools.repeat(None)
    start=time.monotonic(); count=0
    workers=6
    with open(path,'x') as f,ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(kind,)) as pool:
        inputs=zip(queries,bases)
        while count<N:
            batch=list(itertools.islice(inputs,1000))
            assert batch
            for result in pool.map(work,batch,chunksize=8):
                f.write(json.dumps(result,ensure_ascii=False)+'\n')
                count+=1
            f.flush()
            print(kind,count,round(time.monotonic()-start,1),flush=True)
    assert count==N
    dump(FRESH/f'{kind}_runtime.json',dict(queries=count,seconds=time.monotonic()-start,workers=workers,parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,max_child_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))


def score():
    verify_freeze()
    assert (FRESH/'baseline_runtime.json').exists() and (FRESH/'improved_runtime.json').exists()
    if (FRESH/'validation.json').exists(): raise RuntimeError('Already scored')
    q=load(FRESH/'queries.json'); truth=load(FRESH/'truth.json'); n=load(SOURCE_POOL/'index_meta.json')['target_count']
    valid={x[0] for x in __import__('sqlite3').connect(f'file:{SOURCE_POOL/"index.sqlite"}?mode=ro',uri=True).execute('SELECT id FROM records')}
    assert set().union(*map(set,truth.values()))<=valid
    groups={kind:collections.defaultdict(prior.blank) for kind in ['frozen','updated']}
    for kind in groups: (FRESH/kind).mkdir()
    files={kind:open(FRESH/kind/'candidate_pairs.tsv','x') for kind in groups}
    for f in files.values(): f.write('source1_entity_id\tcandidate_entity_ids\n')
    recovered=added=added_false=max_added=0
    try:
        for i,(query,base,raw) in enumerate(zip(q,read_jsonl(FRESH/'baseline.jsonl'),read_jsonl(FRESH/'improved.jsonl')),1):
            sid=query['entity_id']; assert base['id']==raw['id']==sid and base['country']==raw['country']==query['country']
            assert set(raw['baseline'])==set(baseline_set(base))
            old=set(select(raw,OLD_THRESHOLD,OLD_CAP)); new=set(select(raw,NEW_THRESHOLD,NEW_CAP)); true=set(truth[sid])
            assert old<=new<=valid and len(new-old)<=NEW_CAP
            extras=new-old; added+=len(extras); recovered+=len(extras&true); added_false+=len(extras-true); max_added=max(max_added,len(extras))
            for kind,candidates in [('frozen',old),('updated',new)]:
                prior.add(groups[kind]['overall'],true,candidates)
                prior.add(groups[kind][query['country']],true,candidates)
                files[kind].write(sid+'\t'+','.join(sorted(candidates))+'\n')
        assert i==N
    finally:
        for f in files.values(): f.close()
    metrics={kind:dict(overall=prior.finish(groups[kind]['overall'],n),countries={c:prior.finish(groups[kind][c],n) for c in sorted(groups[kind]) if c!='overall'}) for kind in groups}
    for kind in groups: dump(FRESH/kind/'metrics.json',metrics[kind])
    paired=dict(recovered_true_pairs=recovered,lost_true_pairs=0,added_candidates=added,additional_false_candidates=added_false,mean_added_candidates=added/N,max_added_candidates=max_added)
    dump(FRESH/'paired.json',paired)
    assert metrics['updated']['overall']['TP']-metrics['frozen']['overall']['TP']==recovered
    assert metrics['updated']['overall']['FP']-metrics['frozen']['overall']['FP']==added_false
    for kind in groups:
        with open(FRESH/kind/'candidate_pairs.tsv',newline='') as f:
            parsed=csv.DictReader(f,delimiter='\t')
            assert parsed.fieldnames==['source1_entity_id','candidate_entity_ids']
            for query,raw,line in zip(q,read_jsonl(FRESH/'improved.jsonl'),parsed):
                assert query['entity_id']==raw['id']==line['source1_entity_id']
                expected=select(raw,OLD_THRESHOLD,OLD_CAP) if kind=='frozen' else select(raw,NEW_THRESHOLD,NEW_CAP)
                actual=line['candidate_entity_ids'].split(',') if line['candidate_entity_ids'] else []
                assert actual==sorted(set(expected)) and set(actual)<=valid
            assert next(parsed,None) is None
    dump(FRESH/'validation.json',dict(status='PASS',rows=N,target_count=n,all_truth_targets_present=True,exact_exported_sets=True,frozen_subset_updated=True,source_and_input_hashes_unchanged=True))
    print(json.dumps(dict(metrics=metrics,paired=paired),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['development','prepare','freeze','baseline','improved','score'])
    stage=parser.parse_args().stage
    run(stage) if stage in ('baseline','improved') else globals()[stage]()
