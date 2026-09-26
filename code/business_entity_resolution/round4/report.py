"""Human-readable paired validation report for frozen round-4 selection."""
import importlib.util
import pathlib

spec=importlib.util.spec_from_file_location('round4_evaluation_module',pathlib.Path(__file__).with_name('evaluate.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ROOT,ART,FRESH=module.ROOT,module.ART,module.FRESH
NEW_THRESHOLD,NEW_CAP=module.NEW_THRESHOLD,module.NEW_CAP
OLD_THRESHOLD,OLD_CAP=module.OLD_THRESHOLD,module.OLD_CAP
verify_freeze,load=module.verify_freeze,module.load


def table(left, right):
    keys=[('queries','S1 count'),('targets','Target count'),('comparison_space','Comparison space'),('TP','TP'),('FN','FN'),('FP','FP'),('TN','TN'),('recall','Recall'),('precision','Candidate precision'),('reduction_ratio','Reduction ratio'),('specificity','Specificity'),('candidate_f1','Candidate F1'),('candidates','Candidate pairs'),('candidates_mean','Candidates/S1 mean'),('candidates_p95','Candidates/S1 p95'),('candidates_max','Candidates/S1 max'),('oracle_macro_f05','Oracle macro F0.5 ceiling'),('singletons','True singletons'),('entities_with_misses','S1 with at least one miss')]
    pct={'recall','precision','reduction_ratio','specificity','oracle_macro_f05'}
    def val(m,key):
        x=m[key]
        return f'{x*100:.5f}%' if key in pct else f'{x:.6f}' if isinstance(x,float) else f'{x:,}'
    return '\n'.join(['| Metric | Frozen round 2 | Updated selection |','|---|---:|---:|']+[f'| {label} | {val(left,key)} | {val(right,key)} |' for key,label in keys])


def main():
    cfg=verify_freeze()
    assert load(FRESH/'validation.json')['status']=='PASS'
    old=load(FRESH/'frozen/metrics.json')
    new=load(FRESH/'updated/metrics.json')
    paired=load(FRESH/'paired.json')
    rt={k:load(FRESH/f'{k}_runtime.json') for k in ['baseline','improved']}
    dev=load(ART/'development_choice.json')
    im=new['countries']['India']; overall=new['overall']
    met=im['recall']>=.995 and overall['recall']>=.995
    lines=['# Round 4: fresh validation of the blocking selection change','',f"**Updated recall: {overall['recall']*100:.5f}% overall; {im['recall']*100:.5f}% India.** The >=99.5% overall/India criterion was {'met' if met else 'not met'}. This is a 10,000-reference fresh slice scored after the rule was frozen.",'',table(old['overall'],new['overall']),'','## Country-level paired results','']
    for c in sorted(new['countries']):
        lines += [f'### {c}','',table(old['countries'][c],new['countries'][c]),'']
    lines += ['## What changed','',f"The original round-2 candidate rule used threshold {OLD_THRESHOLD:.2f} and at most {OLD_CAP} rescues. The updated rule uses threshold {NEW_THRESHOLD:.2f} and at most {NEW_CAP} rescues from the same frozen retrieval rankings. It retains every original candidate. On this fresh slice it recovered {paired['recovered_true_pairs']:,} true links and lost none, adding {paired['added_candidates']:,} candidates ({paired['mean_added_candidates']:.3f} per S1; maximum {paired['max_added_candidates']}), including {paired['additional_false_candidates']:,} nontrue candidates. A larger candidate set increases downstream matching work and may lower candidate precision.",'','## Development and independent check','',f"Rows 20,001–50,000 were inspected development data; the choice reached {dev['india_recall']*100:.5f}% India recall with {dev['mean_candidates']:.3f} candidates/S1 on that slice. Rows 50,001–60,000 were reserved as the next evaluation slice. The exact selection rule, source code, query/truth inputs, and index hashes were sealed at {cfg['frozen_utc']} before either fresh retrieval pass. No fresh-result tuning followed.",'','The common 409,141-target pool was already constructed for original S1 rows 20,001–80,000. It includes every labeled target for the fresh 10k and deterministic 2% distractors. The same pool and raw retrieval rankings serve both candidate rules. The pool includes positives for other S1 rows and omits most of the 10.32 million target corpus, so these are reduced-pool results; full-corpus and unseen-France behavior remain unverified. Labels were used to construct/audit the pool and to score outputs, not for retrieval. IDs identify records, and country does not filter retrieval.','', '## Validation and metric meaning','', 'Validation PASS: both exports have exactly 10,000 aligned S1 rows; every exported target ID belongs to the sealed pool; all known true targets are present; the updated set contains the original set per reference; and the exports were reopened and compared with the exact selected sets. Frozen code and input hashes matched before both retrieval runs and scoring.','', 'Comparison space = S1 rows × all pool targets, including cross-country pairs. TP/FN are kept/missed true links; FP are nontrue candidates; TN is every other pair. Candidate precision and candidate F1 measure blocking output, not final match decisions. The oracle macro F0.5 ceiling assumes a perfect later classifier that removes every false candidate, and assigns true singletons a score of one for an empty prediction. No final matching classifier or final F0.5 was evaluated.','', '## Runtime and artifacts','']
    for k in ['baseline','improved']:
        r=rt[k]
        lines.append(f"- {k} retrieval: {r['seconds']:.2f}s wall time, {r['workers']} workers, parent peak RSS {r['parent_peak_rss_bytes']/1e6:.1f} MB, maximum child peak RSS {r['max_child_peak_rss_bytes']/1e6:.1f} MB.")
    lines += ['', 'These child peaks do not measure aggregate concurrent memory. Pool construction happened in round 3 and is separate from these retrieval times.','', 'Run details: `code/business_entity_resolution/round4/README.md`. Frozen configuration: `artifacts/blocking_round4/frozen_config.json`. Exact candidate TSVs and metrics: `artifacts/blocking_round4/fresh/frozen/` and `artifacts/blocking_round4/fresh/updated/`. Earlier code and artifacts are preserved.','']
    path=ROOT/'reports/blocking_round4_evaluation.md'
    path.write_text('\n'.join(lines))
    print(path)


if __name__=='__main__':main()
