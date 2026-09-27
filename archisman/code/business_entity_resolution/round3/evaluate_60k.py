"""Frozen round-2 blocker evaluated on original S1 rows 20,001–80,000."""
import argparse
import collections
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib
import itertools
import json
import math
import os
import pathlib
import re
import resource
import sqlite3
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
R2 = ROOT / 'code/business_entity_resolution/round2'
sys.path.insert(0, str(R2))
from common import ART as R2_ART, OLD, SRC, FTS_SQL, dump, fields, load, rows, tokens
from phonetic import grams
from normalize import address, bigrams

ART = ROOT / 'artifacts/blocking_round3'
FRESH = ART / 'fresh'
EVALUATED = 30000
CFG = load(R2_ART / 'frozen_config.json')
SOURCE_FILES = [R2 / x for x in ['common.py', 'normalize.py', 'improve.py', 'run.py', 'extra_index.py']]
SOURCE_FILES += [SRC / x for x in ['blocking.py', 'evaluate.py', 'phonetic.py']]


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify_code():
    assert {str(p.relative_to(ROOT)): digest(p) for p in SOURCE_FILES} == CFG['code_hashes']
    assert digest(OLD / 'frozen_config.json') == CFG['baseline_config_sha256']
    assert CFG['cap'] == 4 and CFG['threshold'] == .7


def verify_inputs():
    verify_code()
    sealed = load(ART / 'sealed.json')
    for rel, sha in sealed['input_hashes'].items():
        assert digest(ROOT / rel) == sha, f'Frozen input changed: {rel}'
    assert sealed['config_sha256'] == digest(R2_ART / 'frozen_config.json')


def prepare():
    verify_code()
    if FRESH.exists():
        raise RuntimeError('Round-3 evaluation already exists')
    FRESH.mkdir(parents=True)
    start = time.monotonic()
    prior = list(itertools.islice(rows(ROOT / 'dataset/train/train_source1.tsv'), 20000))
    queries = list(itertools.islice(rows(ROOT / 'dataset/train/train_source1.tsv'), 20000, 80000))
    assert len(prior) == 20000 and len(queries) == 60000
    prior_ids = {q['entity_id'] for q in prior}
    ids = {q['entity_id'] for q in queries}
    assert len(prior_ids) == 20000 and len(ids) == 60000 and not (prior_ids & ids)
    sig = lambda q: (tuple(fields(q)[:2]), q['country'])
    signature_overlap = len({sig(q) for q in prior} & {sig(q) for q in queries})
    truth = {}
    owner = {}
    prior_targets = set()
    audited_rows = 0
    for r in rows(ROOT / 'dataset/train/train_ground_truth.tsv'):
        audited_rows += 1
        sid = r['source1_entity_id']
        mids = r['matched_entity_ids'].split(',') if r['matched_entity_ids'] else []
        if sid in prior_ids:
            prior_targets.update(mids)
        if sid in ids:
            assert sid not in truth
            truth[sid] = mids
            for mid in mids:
                assert mid not in owner, 'Target shared by selected groups'
                owner[mid] = sid
    assert len(truth) == 60000 and not (prior_targets & owner.keys())
    # A second full pass checks ownership outside all selected groups.
    for r in rows(ROOT / 'dataset/train/train_ground_truth.tsv'):
        sid = r['source1_entity_id']
        for mid in (r['matched_entity_ids'].split(',') if r['matched_entity_ids'] else []):
            if mid in owner:
                assert owner[mid] == sid, 'Selected target owned by another S1 group'
    dump(FRESH / 'queries.json', queries)
    dump(FRESH / 'truth.json', truth)
    required = set(owner)
    db = sqlite3.connect(FRESH / 'index.sqlite')
    db.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; CREATE TABLE records(id TEXT PRIMARY KEY,name TEXT,address TEXT,country TEXT);' + FTS_SQL)
    batch, fbatch = [], []
    n = scanned = 0
    countries = collections.Counter()
    cutoff = int(.02 * 2**32)
    for source in [2, 3]:
        for r in rows(ROOT / f'dataset/train/train_source{source}.tsv'):
            scanned += 1
            mid = r['entity_id']
            if mid not in required and int.from_bytes(hashlib.sha256(('distractor-v1' + mid).encode()).digest()[:4], 'big') >= cutoff:
                continue
            n += 1
            countries[r['country']] += 1
            batch.append((n, mid, r['business_name'], r['business_address'], r['country']))
            fbatch.append((n, *fields(r), ' '.join(grams(r['business_name']))))
            if len(batch) == 10000:
                db.executemany('INSERT INTO records(rowid,id,name,address,country) VALUES(?,?,?,?,?)', batch)
                db.executemany('INSERT INTO search(rowid,name,address,grams,phonetic) VALUES(?,?,?,?,?)', fbatch)
                db.commit()
                batch, fbatch = [], []
    if batch:
        db.executemany('INSERT INTO records(rowid,id,name,address,country) VALUES(?,?,?,?,?)', batch)
        db.executemany('INSERT INTO search(rowid,name,address,grams,phonetic) VALUES(?,?,?,?,?)', fbatch)
    db.execute('CREATE VIRTUAL TABLE vocab USING fts5vocab(search,col)')
    db.commit()
    found = {r[0] for r in db.execute('SELECT id FROM records WHERE id IN (SELECT id FROM records)')}
    assert required <= found and len(found) == n
    db.close()
    dump(FRESH / 'index_meta.json', dict(target_count=n, required_targets=len(required), distractors=n-len(required), scanned_targets=scanned, countries=countries, seconds=time.monotonic()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    dump(FRESH / 'manifest.json', dict(rows='20001 through 80000 inclusive in original S1 order', reference_count=60000, truth_rows_audited=audited_rows, shared_s1_ids=0, shared_labeled_targets=0, exact_normalized_name_address_country_overlap=signature_overlap, target_policy='All labeled positives plus deterministic 2% SHA256(distractor-v1 + target ID) sample; no country restriction', configuration='Frozen round-2 threshold 0.70 and cap 4; no evaluation-batch tuning', full_target_corpus=scanned))
    print('Prepared 60k benchmark', n, 'targets;', len(required), 'positives;', signature_overlap, 'exact prior signatures', flush=True)


def extra_index():
    verify_code()
    out = FRESH / 'extra.sqlite'
    if out.exists():
        raise RuntimeError('Extra index exists')
    src = sqlite3.connect(f'file:{FRESH / "index.sqlite"}?mode=ro', uri=True)
    db = sqlite3.connect(out)
    start = time.monotonic()
    db.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; CREATE VIRTUAL TABLE extra USING fts5(native,addr,n2,content="",detail=full);')
    batch = []
    for rid, name, addr in src.execute('SELECT rowid,name,address FROM records'):
        batch.append((rid, ' '.join(tokens(name)), ' '.join(address(addr)), ' '.join(sorted(bigrams(name)))))
        if len(batch) == 10000:
            db.executemany('INSERT INTO extra(rowid,native,addr,n2) VALUES(?,?,?,?)', batch)
            batch = []
    if batch:
        db.executemany('INSERT INTO extra(rowid,native,addr,n2) VALUES(?,?,?,?)', batch)
    db.execute('CREATE VIRTUAL TABLE vocab USING fts5vocab(extra,col)')
    db.commit()
    db.close()
    src.close()
    dump(FRESH / 'extra_meta.json', dict(seconds=time.monotonic()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, bytes=out.stat().st_size))
    print('Extra index complete', flush=True)


def seal():
    verify_code()
    if (ART / 'sealed.json').exists():
        raise RuntimeError('Already sealed')
    names = ['queries.json', 'truth.json', 'index.sqlite', 'extra.sqlite', 'manifest.json']
    assert all((FRESH / x).exists() for x in names)
    dump(ART / 'sealed.json', dict(config_sha256=digest(R2_ART / 'frozen_config.json'), code_hashes=CFG['code_hashes'], input_hashes={str((FRESH / x).relative_to(ROOT)): digest(FRESH / x) for x in names}, sealed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print('Sealed inputs and frozen code', flush=True)


ENGINE = None


def init_worker(kind):
    global ENGINE
    if kind == 'baseline':
        ev = importlib.import_module('evaluate')
        ev.ART = FRESH
        ENGINE = ev.Retriever()
    else:
        imp = importlib.import_module('improve')
        imp.ART = ART
        ENGINE = imp.Improved('fresh')


def retrieve(item):
    q, base = item
    return ENGINE.retrieve(q) if base is None else ENGINE.retrieve(q, base)


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def follow_baseline():
    deadline = time.monotonic() + 14400
    with open(FRESH / 'baseline.jsonl') as f:
        count = 0
        while count < EVALUATED:
            line = f.readline()
            if line:
                count += 1
                yield json.loads(line)
            elif (FRESH / 'baseline_runtime.json').exists():
                raise RuntimeError(f'Baseline finished with only {count} results')
            elif time.monotonic() > deadline:
                raise TimeoutError(f'Baseline stalled after {count} results')
            else:
                time.sleep(.5)


def run(kind):
    verify_inputs()
    assert kind in ('baseline', 'improved')
    output = FRESH / f'{kind}.jsonl'
    if output.exists():
        raise RuntimeError(f'{kind} output exists')
    if kind == 'improved':
        assert (FRESH / 'baseline.jsonl').exists()
    queries = load(FRESH / 'queries.json')[:EVALUATED]
    assert len(queries) == EVALUATED
    baselines = follow_baseline() if kind == 'improved' else itertools.repeat(None)
    start = time.monotonic()
    count = 0
    workers = 6 if kind == 'improved' else 3
    with open(output, 'x') as f, ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(kind,)) as pool:
        inputs = zip(queries, baselines)
        while count < len(queries):
            batch = list(itertools.islice(inputs, 1000))
            assert batch, f'Only {count} retrieval inputs available'
            for result in pool.map(retrieve, batch, chunksize=8):
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                count += 1
            f.flush()
            print(kind, count, f'{time.monotonic()-start:.1f}s', flush=True)
    assert count == EVALUATED
    dump(FRESH / f'{kind}_runtime.json', dict(queries=count, seconds=time.monotonic()-start, parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, max_child_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, workers=workers))
    print(kind, 'complete', flush=True)


def record_stop():
    verify_inputs()
    if (FRESH / 'stopped_at_30k.json').exists():
        raise RuntimeError('Stop marker already exists')
    raw_counts = {}
    for kind, workers in [('baseline', 3), ('improved', 6)]:
        log = (FRESH / f'{kind}.log').read_text()
        match = re.search(rf'^{kind} {EVALUATED} ([0-9.]+)s$', log, re.MULTILINE)
        assert match, f'{kind} did not reach {EVALUATED}'
        count = sum(1 for _ in read_jsonl(FRESH / f'{kind}.jsonl'))
        assert count >= EVALUATED
        raw_counts[kind] = count
        runtime = FRESH / f'{kind}_runtime.json'
        if not runtime.exists():
            dump(runtime, dict(queries=EVALUATED, seconds=float(match.group(1)), workers=workers, time_source='progress log at row 30000', parent_peak_rss_bytes=None, max_child_peak_rss_bytes=None, interrupted_at_user_request=True))
    dump(FRESH / 'stopped_at_30k.json', dict(evaluated=EVALUATED, original_s1_rows='20001–50000 inclusive', pool_built_for_original_s1_rows='20001–80000 inclusive', raw_retrieval_rows=raw_counts, extra_raw_rows_unscored=any(x>EVALUATED for x in raw_counts.values()), stop_reason='User requested stop at 30k', recorded_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print('Recorded 30k stop', raw_counts, flush=True)


def blank():
    return dict(TP=0, FN=0, FP=0, sizes=[], oracle_sum=0., entities_with_misses=0, singletons=0, queries=0)


def add(acc, truth, cand):
    hit = len(truth & cand)
    acc['TP'] += hit
    acc['FN'] += len(truth)-hit
    acc['FP'] += len(cand)-hit
    acc['sizes'].append(len(cand))
    acc['oracle_sum'] += 1. if not truth else (1.25*hit/(hit+.25*len(truth)) if hit else 0.)
    acc['entities_with_misses'] += bool(truth-cand)
    acc['singletons'] += not truth
    acc['queries'] += 1


def finish(acc, target_count):
    q = acc['queries']
    tp, fn, fp = (acc[x] for x in ('TP', 'FN', 'FP'))
    space = q*target_count
    tn = space-tp-fn-fp
    sizes = sorted(acc['sizes'])
    recall = tp/(tp+fn) if tp+fn else 1.
    return dict(queries=q, targets=target_count, comparison_space=space, TP=tp, FN=fn, FP=fp, TN=tn, recall=recall, fn_rate=1-recall, precision=tp/(tp+fp), reduction_ratio=1-(tp+fp)/space, specificity=tn/(tn+fp), candidate_f1=2*tp/(2*tp+fp+fn), candidates=tp+fp, candidates_mean=statistics.mean(sizes), candidates_p95=sizes[math.ceil(.95*q)-1], candidates_max=sizes[-1], oracle_macro_f05=acc['oracle_sum']/q, entities_with_misses=acc['entities_with_misses'], singletons=acc['singletons'])


def score():
    verify_inputs()
    assert load(FRESH / 'stopped_at_30k.json')['evaluated'] == EVALUATED
    assert all((FRESH / f'{k}_runtime.json').exists() for k in ['baseline', 'improved'])
    if (FRESH / 'validation.json').exists():
        raise RuntimeError('Already scored')
    imp = importlib.import_module('improve')
    queries = load(FRESH / 'queries.json')[:EVALUATED]
    all_truth = load(FRESH / 'truth.json')
    truth = {q['entity_id']:all_truth[q['entity_id']] for q in queries}
    n = load(FRESH / 'index_meta.json')['target_count']
    valid = {r[0] for r in sqlite3.connect(f'file:{FRESH / "index.sqlite"}?mode=ro', uri=True).execute('SELECT id FROM records')}
    assert set().union(*map(set, truth.values())) <= valid
    stats = {k: collections.defaultdict(blank) for k in ['baseline', 'improved']}
    for kind in ['baseline', 'improved']:
        (FRESH / kind).mkdir(exist_ok=True)
    outputs = {k: open(FRESH / k / 'candidate_pairs.tsv', 'x') for k in ['baseline', 'improved']}
    for path in outputs.values():
        path.write('source1_entity_id\tcandidate_entity_ids\n')
    recovered = added = added_false = max_added = 0
    try:
        for i, (q, b, r) in enumerate(zip(queries, read_jsonl(FRESH / 'baseline.jsonl'), read_jsonl(FRESH / 'improved.jsonl')), 1):
            sid = q['entity_id']
            assert b['id'] == r['id'] == sid and b['country'] == r['country'] == q['country']
            base = set(imp.baseline_set(b))
            enhanced = set(imp.select(r, CFG['threshold'], CFG['cap']))
            assert base <= enhanced and enhanced <= valid and base <= valid
            assert set(r['baseline']) == base
            assert len(enhanced-base) <= CFG['cap']
            true = set(truth[sid])
            extras = enhanced-base
            added += len(extras)
            added_false += len(extras-true)
            recovered += len(extras & true)
            max_added = max(max_added, len(extras))
            for kind, cand in [('baseline', base), ('improved', enhanced)]:
                add(stats[kind]['overall'], true, cand)
                add(stats[kind][q['country']], true, cand)
                outputs[kind].write(sid + '\t' + ','.join(sorted(cand)) + '\n')
            if i % 10000 == 0:
                print('Scored', i, flush=True)
        assert i == len(queries) == EVALUATED
        for stream in (read_jsonl(FRESH / 'baseline.jsonl'), read_jsonl(FRESH / 'improved.jsonl')):
            assert sum(1 for _ in stream) >= EVALUATED
    finally:
        for path in outputs.values():
            path.close()
    result = {kind: dict(overall=finish(group['overall'], n), countries={c:finish(group[c], n) for c in sorted(group) if c != 'overall'}) for kind, group in stats.items()}
    for kind in ['baseline', 'improved']:
        dump(FRESH / kind / 'metrics.json', result[kind])
    paired = dict(recovered_true_pairs=recovered, lost_true_pairs=0, added_candidates=added, mean_added_candidates=added/EVALUATED, max_added_candidates=max_added, additional_false_candidates=added_false)
    dump(FRESH / 'paired.json', paired)
    assert result['improved']['overall']['TP']-result['baseline']['overall']['TP'] == recovered
    assert result['improved']['overall']['FP']-result['baseline']['overall']['FP'] == added_false
    assert all(result[k]['overall']['TP']+result[k]['overall']['FN'] == len(set().union(*map(set, truth.values()))) for k in result)
    # Reopen both exports and compare every row with the frozen selected sets.
    for kind in ['baseline', 'improved']:
        with open(FRESH / kind / 'candidate_pairs.tsv', newline='') as f:
            parsed = csv.DictReader(f, delimiter='\t')
            assert parsed.fieldnames == ['source1_entity_id', 'candidate_entity_ids']
            count = 0
            source = read_jsonl(FRESH / f'{kind}.jsonl')
            for q, record, line in zip(queries, source, parsed):
                sid = q['entity_id']
                assert line['source1_entity_id'] == record['id'] == sid
                actual = line['candidate_entity_ids'].split(',') if line['candidate_entity_ids'] else []
                expected = imp.baseline_set(record) if kind == 'baseline' else imp.select(record, CFG['threshold'], CFG['cap'])
                assert actual == sorted(set(expected)) and set(actual) <= valid
                count += 1
            assert count == EVALUATED and next(parsed, None) is None
    dump(FRESH / 'validation.json', dict(status='PASS', source1_rows=EVALUATED, targets=n, all_true_targets_present=True, exact_exported_sets=True, baseline_is_subset=True, code_and_data_hashes_unchanged=True, partial_retrieval_files_have_extra_unscored_rows=True, result_files={k:digest(FRESH / k / 'candidate_pairs.tsv') for k in outputs}))
    print(json.dumps(dict(metrics=result, paired=paired), indent=2), flush=True)


def report():
    verify_inputs()
    assert load(FRESH / 'validation.json')['status'] == 'PASS'
    metrics = {k:load(FRESH / k / 'metrics.json') for k in ('baseline', 'improved')}
    paired = load(FRESH / 'paired.json')
    pool = load(FRESH / 'index_meta.json')
    manifest = load(FRESH / 'manifest.json')
    sealed = load(ART / 'sealed.json')
    runtime = {k:load(FRESH / f'{k}_runtime.json') for k in ('baseline', 'improved')}
    extra = load(FRESH / 'extra_meta.json')
    labels = [('queries','S1 count'), ('targets','Target count'), ('comparison_space','Comparison space'), ('TP','TP'), ('FN','FN'), ('FP','FP'), ('TN','TN'), ('recall','Recall'), ('precision','Precision'), ('reduction_ratio','Reduction ratio'), ('specificity','Specificity'), ('candidate_f1','Candidate F1'), ('candidates','Candidate pairs'), ('candidates_mean','Candidates/S1 mean'), ('candidates_p95','Candidates/S1 p95'), ('candidates_max','Candidates/S1 max'), ('oracle_macro_f05','Oracle macro F0.5 ceiling'), ('singletons','True singletons'), ('entities_with_misses','S1 with at least one miss')]
    pct = {'recall','precision','reduction_ratio','specificity','oracle_macro_f05'}
    def table(left, right):
        lines = ['| Metric | Original baseline | Frozen improvement |','|---|---:|---:|']
        for key, label in labels:
            def fmt(m):
                value = m[key]
                return f'{value*100:.5f}%' if key in pct else f'{value:.6f}' if isinstance(value,float) else f'{value:,}'
            lines.append(f'| {label} | {fmt(left)} | {fmt(right)} |')
        return '\n'.join(lines)
    a, b = (metrics[k]['overall'] for k in ('baseline','improved'))
    india = metrics['improved']['countries'].get('India')
    assert india
    met = b['recall'] >= .995 and india['recall'] >= .995
    lines = ['# Frozen blocking evaluation: next 30,000 S1 records','',f"**Improved recall: {b['recall']*100:.5f}% overall and {india['recall']*100:.5f}% India.** The >=99.5% overall/India criterion was {'met' if met else 'not met'}. The baseline was run on the identical target pool.",'',table(a,b),'','## India and US','']
    for country in sorted(metrics['baseline']['countries']):
        lines += [f'### {country}','',table(metrics['baseline']['countries'][country],metrics['improved']['countries'][country]),'']
    selected_positive_count = a['TP'] + a['FN']
    lines += ['## Paired change','',f"The improved set recovered {paired['recovered_true_pairs']:,} additional true pairs and lost {paired['lost_true_pairs']:,} baseline true pairs. It added {paired['added_candidates']:,} candidates ({paired['mean_added_candidates']:.4f} per S1, maximum {paired['max_added_candidates']}), including {paired['additional_false_candidates']:,} false candidates. Its per-S1 set is checked as a superset of the original baseline.",'','## Population and audit','',f"Following the user's stop instruction, the evaluated batch is original S1 file rows 20,001–50,000 inclusive: exactly 30,000 references with {selected_positive_count:,} known true links. The pool had already been built for an originally planned 60k run and contains all {pool['required_targets']:,} labeled targets for rows 20,001–80,000 plus {pool['distractors']:,} deterministic 2% distractors: {pool['target_count']:,} total. Targets labeled to the unevaluated second half remain in the common pool as nontrue candidates for these 30k references. Both pipelines searched this same pool. The paired comparison space is 30,000 × {pool['target_count']:,} = {b['comparison_space']:,} pairs. The complete S2/S3 corpus has {pool['scanned_targets']:,} targets.",'',f"The full truth ownership audit found zero shared labeled targets with any other S1 group, including prior batches. S1 IDs do not overlap. The first 20,000 S1 rows and the originally planned 60k batch share {manifest['exact_normalized_name_address_country_overlap']} exact normalized name/address/country signatures. This exact check does not resolve all near-duplicate or labeling ambiguities.",'','The pool includes every known positive but omits most full-corpus negatives, so absolute performance may be optimistic relative to all 10.32 million targets. The 30k result is tied to this 409,141-target pool and should not be compared directly with the earlier 10k pools. Consecutive file rows may not represent unseen data. France accuracy remains unmeasured. Country values were never used to filter retrieval. These are blocking candidate metrics, not final matching precision, F1, F0.5, or leaderboard scores.','', '## Frozen method and validation','',f"The round-2 source and settings were reused without modification: rescue score threshold {CFG['threshold']:.2f}, at most {CFG['cap']} additions per S1. Original code/configuration hashes were checked before preparation and each retrieval. The new query, truth, index, extra index, and manifest hashes were sealed at {sealed['sealed_utc']} before running either pipeline. No selection or tuning used this batch.",'','Labels were used only to build and audit the pool and to score the completed slice; IDs identify rows and seed the deterministic sample. Both exported TSVs were reopened and checked row by row against the exact frozen selected sets and target IDs. All known positives for these 30k references are in the pool, the two scored runs contain exactly 30,000 ordered references, and every original candidate is preserved. The raw retrieval JSONL files retain some unscored rows written during shutdown. Validation status: PASS.','', '## Metric definitions','', 'For each group, comparison space = S1 count × entire target pool, including cross-country pairs. TP/FN are retained/missed true links; FP counts nontrue candidates; TN is comparison space minus TP, FN, and FP. Recall = TP/(TP+FN); precision = TP/(TP+FP); reduction ratio = 1−(TP+FP)/comparison space; specificity = TN/(TN+FP); candidate F1 = 2TP/(2TP+FP+FN). p95 uses the nearest-rank percentile.','', 'The oracle macro F0.5 ceiling assumes a perfect downstream classifier discards every false candidate. It averages per-S1 F0.5 and gives true singletons a score of one when predicting an empty set. It is an upper bound, not an achieved classifier score.','', '## Runtime and memory','',f"Pool construction and ownership audit: {pool['seconds']:.2f}s, parent peak RSS {pool['peak_rss_bytes']/1e6:.1f} MB. Extra index construction: {extra['seconds']:.2f}s, parent peak RSS {extra['peak_rss_bytes']/1e6:.1f} MB.",'']
    for kind in ('baseline','improved'):
        r=runtime[kind]
        lines.append(f"- {kind}: {r['seconds']:.1f}s to the 30,000th result, measured from the progress log; {r['workers']} workers. Retrieval RSS was not captured because the processes were interrupted at the requested stopping point.")
    lines += ['', 'The two retrieval processes overlapped in wall time, so their durations should not be added to estimate elapsed time. Input preparation, indexing, scoring, and report generation are separate.','','## Artifacts','', 'Reproduction commands: `code/business_entity_resolution/round3/README.md`. Frozen configuration: `artifacts/blocking_round2/frozen_config.json`; new input seal: `artifacts/blocking_round3/sealed.json`. Candidate TSVs and metrics: `artifacts/blocking_round3/fresh/baseline/` and `artifacts/blocking_round3/fresh/improved/`. Prior round artifacts remain unchanged.','']
    path = ROOT / 'reports/blocking_round3_30k_evaluation.md'
    path.write_text('\n'.join(lines))
    print(path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'extra_index', 'seal', 'baseline', 'improved', 'record_stop', 'score', 'report'])
    stage = parser.parse_args().stage
    if stage in ('baseline', 'improved'):
        run(stage)
    else:
        globals()[stage]()
