"""Score both frozen candidate sets only after both retrieval runs finish."""
import csv,sqlite3,importlib.util,collections,hashlib
from common import *
from run import read_lines,check_freeze,metrics
from improve import select,baseline_set

def generate():
    cfg=check_freeze()
    assert load(ART/'FRESH_OPENED.json')['config_sha256']==hashlib.sha256((ART/'frozen_config.json').read_bytes()).hexdigest(), 'Configuration changed after opening'
    n=load(ART/'fresh/index_meta.json')['target_count']
    assert (ART/'fresh/baseline_runtime.json').exists() and (ART/'fresh/improved_runtime.json').exists()
    br=list(read_lines(ART/'fresh/baseline.jsonl')); ir=list(read_lines(ART/'fresh/improved.jsonl'))
    queries=load(ART/'fresh/queries.json'); querymap={q['entity_id']:q for q in queries}
    assert len(br)==len(ir)==len(queries)==10000
    assert {r['id'] for r in br}=={r['id'] for r in ir}==set(querymap)
    truth=load(ART/'fresh/truth.json'); db=sqlite3.connect(f'file:{ART}/fresh/index.sqlite?mode=ro',uri=True)
    valid={r[0] for r in db.execute('SELECT id FROM records')}; assert set().union(*map(set,truth.values()))<=valid
    bs={r['id']:set(baseline_set(r)) for r in br}; improved={r['id']:set(select(r,cfg['threshold'],cfg['cap'])) for r in ir}
    assert all(bs[s]<=improved[s] for s in bs), 'Baseline lost a candidate'
    ms={'baseline':metrics(br,truth,n,kind='baseline'),'improved':metrics(ir,truth,n,cfg['threshold'],cfg['cap'])}
    spec=importlib.util.spec_from_file_location('validator',ROOT/'utils/validate_submission.py'); validator=importlib.util.module_from_spec(spec); spec.loader.exec_module(validator)
    for kind,mapping in [('baseline',bs),('improved',improved)]:
        out=ART/'fresh'/kind; out.mkdir(exist_ok=True)
        dump(out/'metrics.json',ms[kind])
        with open(out/'candidate_pairs.tsv','w') as f:
            f.write('source1_entity_id\tcandidate_entity_ids\n')
            for q in queries: f.write(q['entity_id']+'\t'+','.join(sorted(mapping[q['entity_id']]))+'\n')
        errors=[]
        parsed=validator.validate_id_list_file(str(out/'candidate_pairs.tsv'),validator.CANDIDATE_HEADER,'candidate_entity_ids',set(querymap),valid,errors)
        assert not errors and parsed==mapping, errors
        ranked_by_id={r['id']:r['ranked'] for r in (br if kind=='baseline' else ir)}
        misses=[]
        for q in queries:
            for mid in sorted(set(truth[q['entity_id']])-mapping[q['entity_id']]):
                target=db.execute('SELECT id,name,address,country FROM records WHERE id=?',(mid,)).fetchone()
                ranks=ranked_by_id[q['entity_id']]
                misses.append(dict(query=q,target=target,retrieved_rank=ranks.index(mid)+1 if mid in ranks else None))
        dump(out/'misses.json',misses)
    additions=[len(improved[s]-bs[s]) for s in bs]
    paired=dict(recovered_true_pairs=ms['improved']['overall']['TP']-ms['baseline']['overall']['TP'],lost_true_pairs=0,added_candidates=sum(additions),mean_added_candidates=sum(additions)/len(additions),max_added_candidates=max(additions),additional_false_candidates=ms['improved']['overall']['FP']-ms['baseline']['overall']['FP'])
    dump(ART/'fresh/paired.json',paired)
    dump(ART/'fresh/validation.json',dict(status='PASS',source1_rows=10000,targets=n,all_true_targets_present=True,exact_exported_sets=True,baseline_is_subset=True,code_and_data_hashes_unchanged=True))
    write_report(ms,paired,cfg)
    print(json.dumps({'metrics':ms,'paired':paired},indent=2))

def table(cols):
    keys=[('queries','S1 count'),('targets','Target count'),('comparison_space','Comparison space'),('TP','TP'),('FN','FN'),('FP','FP'),('TN','TN'),('recall','Recall'),('fn_rate','FN rate'),('precision','Precision'),('reduction_ratio','Reduction ratio'),('specificity','Specificity'),('candidate_f1','Candidate F1'),('candidates','Candidate pairs'),('candidates_mean','Candidates/S1 mean'),('candidates_p95','Candidates/S1 p95'),('candidates_max','Candidates/S1 max'),('oracle_macro_f05','Oracle macro F0.5 ceiling'),('singletons','True singletons'),('entities_with_misses','S1 with ≥1 miss')]
    pct={'recall','fn_rate','precision','reduction_ratio','specificity','oracle_macro_f05'}
    out=['| Metric | '+' | '.join(cols)+' |','|---|'+'---:|'*len(cols)]
    for key,label in keys:
        vals=[]
        for m in cols.values():
            x=m[key]; vals.append(f'{x*100:.5f}%' if key in pct else f'{x:.6f}' if isinstance(x,float) else f'{x:,}')
        out.append('| '+label+' | '+' | '.join(vals)+' |')
    return '\n'.join(out)
def write_report(ms,paired,cfg):
    b=ms['baseline']['overall']; m=ms['improved']['overall']; india=ms['improved']['countries'].get('India',{}); rt={k:load(ART/'fresh'/f'{k}_runtime.json') for k in ['baseline','improved']}; prep=load(ART/'fresh/index_meta.json'); extra=load(ART/'fresh/extra_meta.json')
    lines=['# Round 2: paired blocking evaluation on the next 10,000 references','',f"**Fresh overall recall: {m['recall']*100:.5f}%; India: {india.get('recall',0)*100:.5f}%.** The >=99.5% target was {'met' if m['recall']>=.995 and india.get('recall',0)>=.995 else 'not met in all requested groups'}. The original baseline was evaluated on the same 239,812-target pool. No fresh-result tuning followed.",'',table({'Original baseline':b,'Improved':m}),'','## Country-level paired results','']
    for country in ms['baseline']['countries']:
        lines += [f'### {country}','',table({'Original baseline':ms['baseline']['countries'][country],'Improved':ms['improved']['countries'][country]}),'']
    lines += ['## Paired change','',f"The improvement recovered {paired['recovered_true_pairs']} true pairs and lost zero baseline pairs. It added {paired['added_candidates']:,} candidate pairs ({paired['mean_added_candidates']:.3f}/S1 on average; maximum {paired['max_added_candidates']}), including {paired['additional_false_candidates']:,} additional false candidates. Superset preservation is checked for every reference; it guarantees paired recall cannot decline, while candidate cost can increase.",'','## Development and freeze','',f"All first 10,000 records—including the former inspected holdout—were development data. The chosen rescue threshold is {cfg['threshold']:.2f}, with at most {cfg['cap']} added candidates per reference. Development selection examined caps 4/8/12/16/24 and cutoffs 0.50–0.80, seeking the smallest mean set with >=99.70% India and overall recall. The complete grid and chosen metrics are saved. Code, baseline configuration, fresh queries, both indices, and the manifest were hashed before fresh evaluation ({cfg['frozen_utc']}).",'','The original baseline source and configuration remain unchanged. New routes use coarse phonetic bigrams with address context; conventional address abbreviations, ordinal and leading-zero normalization; and up to three strongly linked text-derived anchors. Normalization retains original strings in the records. Extra retrieval is capped at 40 normalized-address results, 40 phonetic/address results, and 20 per anchor. Existing filtered baseline retrieval records may also be reconsidered. The final rescue cap governs the actual extra matcher inputs; raw retrieval is not the reported candidate set.','', '## Evaluation population and leakage audit','',f"Fresh queries are original training S1 rows 10,001–20,000. All {prep['required_targets']:,} labeled targets are included alongside {prep['distractors']:,} deterministic distractors drawn with the same 2% hash rule as round 1. Both pipelines use the identical {prep['target_count']:,}-target pool. Labels only construct/audit the benchmark and score outputs; IDs only identify records and reproducible samples. Neither is a retrieval feature.",'','A full truth audit found no overlapping S1 IDs or labeled target ownership between the two batches or another S1 group. No exact normalized name/address/country signature was shared between batches. Shared unlabelled distractors are deliberate. This audit does not independently establish the absence of every possible near-duplicate beyond supplied truth.','', 'The pool is enriched for positives and omits most distractors, so results are optimistic relative to all 10,320,219 targets. Consecutive-row sampling can be biased. France remains untested; all country labels remain eligible, but compatibility is not measured France accuracy. No final classifier, final matching score, or leaderboard result is claimed.','', '## Metric definitions','', 'For each group, N = S1 count × full fresh pool target count (including cross-country pairs). TP/FN count retained/missed true links; FP counts nontrue candidates; TN = N−TP−FN−FP. Recall=TP/(TP+FN); precision=TP/(TP+FP); reduction ratio=1−(TP+FP)/N; specificity=TN/(TN+FP); candidate F1=2TP/(2TP+FP+FN). p95 uses the nearest-rank percentile.','', 'The oracle discards all false candidates. A non-singleton with h retained true links of t scores 1.25h/(h+0.25t), or zero if h=0. Singletons receive an empty set and score one. Averaging these per-S1 scores yields a ceiling, not achieved classifier performance.','', '## Runtime and memory','',f"Fresh pool preparation (ownership audit plus streaming pool/index construction): {prep['seconds']:.2f}s, peak process RSS {prep['peak_rss_bytes']/1e6:.1f} MB. Extra index: {extra['seconds']:.2f}s, peak process RSS {extra['peak_rss_bytes']/1e6:.1f} MB.",'']
    for k,r in rt.items(): lines.append(f"- {k}: {r['seconds']:.2f}s retrieval wall time; parent peak RSS {r['parent_peak_rss_bytes']/1e6:.1f} MB; maximum child peak RSS {r['max_child_peak_rss_bytes']/1e6:.1f} MB; three workers.")
    lines += ['',f"The improved pipeline reuses baseline retrieval and costs {rt['baseline']['seconds']+rt['improved']['seconds']:.2f}s combined retrieval wall time, not merely the incremental rescue time. Input JSON loading precedes the retrieval timer (baseline {rt['baseline'].get('input_load_seconds',0):.2f}s; rescue {rt['improved'].get('input_load_seconds',0):.2f}s) and is excluded from that sum; index construction, scoring, validation, and report generation are separate. RSS is macOS resource usage in bytes; parent and maximum child peaks are not aggregate concurrent memory. These measurements do not establish full-corpus scalability.",'','## Artifacts and checks','', 'Runnable commands: `code/business_entity_resolution/round2/README.md`. Frozen configuration and input hashes: `artifacts/blocking_round2/frozen_config.json`. Fresh candidate TSVs and complete metrics/missed-pair text: `artifacts/blocking_round2/fresh/baseline/` and `.../improved/`. The reports use exactly the final selected sets. Original first-round artifacts remain under `artifacts/blocking/`.','', 'The organizer validator helper checked all 10,000 rows for both candidate sets against the exact pool IDs. Additional checks verify positive inclusion, identical query sets, original-candidate preservation, exact exported sets, and frozen code/data hashes. Unit tests cover metric arithmetic, singleton oracle behavior, score/cap/rescue selection, FTS ranking, Unicode/number normalization, and original-code preservation. These are local evaluation outputs, not competition test submissions.','']
    misses=load(ART/'fresh/improved/misses.json'); summary=dict(total=len(misses),not_retrieved=sum(x['retrieved_rank'] is None for x in misses),filtered=sum(x['retrieved_rank'] is not None for x in misses),empty_target_address=sum(not x['target'][2].strip() for x in misses),indic_script_target_name=sum(any('\u0900'<=c<='\u0dff' for c in x['target'][1]) for x in misses),countries=dict(collections.Counter(x['query']['country'] for x in misses)))
    dump(ART/'fresh/improved/error_summary.json',summary)
    lines += ['## Remaining errors','',f"There are {len(misses)} remaining missed true pairs: {summary['not_retrieved']} were absent from the expanded retrieval and {summary['filtered']} were filtered by final selection. {summary['empty_target_address']} have empty target addresses and {summary['indic_script_target_name']} have Indic-script target names (overlapping categories). Original text for every miss is saved for diagnosis. These fresh outcomes were inspected only after freezing, and were not used for further tuning.",'']
    lines += ['Observed examples include a common name with no address (Surgical Group → surgical group Enterprises, ranked sixth and outside the four-addition cap); a substantially changed name and no address (Global Institute → GLOBAL CENTER, absent from retrieval); and a random-looking trade alias with only partial city/number evidence (Vidhi Medical Centre Clinic → Vioaviavi, retrieved but below the score cutoff). Cross-script names with very short or changed addresses also remain. These are diagnoses only; thresholds and routes were not revised using them.','']
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/blocking_round2_evaluation.md').write_text('\n'.join(lines))
if __name__=='__main__': generate()
