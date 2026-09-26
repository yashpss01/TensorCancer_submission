"""Development-only candidate-size tuning. Never reads holdout labels or rankings."""
import hashlib, json, pathlib
from blocking import ART, dump
from evaluate import measure, export

queries=json.loads((ART/'dev/rankings.json').read_text())
truth=json.loads((ART/'dev_truth.json').read_text())
n=json.loads((ART/'index_meta.json').read_text())['target_count']
results=[]
for k in [10,15,20,30,40,60,80,120,160]:
    for threshold in [0.,.2,.3,.35,.4,.45,.5,.55,.6,.65,.7]:
        m=measure(queries,truth,k,n,threshold=threshold)
        results.append(dict(k=k,threshold=threshold,minimum=0,**m))
dump(ART/'dev/budget_sweep.json',results)
# Predeclared development margin above the requested >=99.5% target.
feasible=[r for r in results if r['recall']>=.9975]
if not feasible:
    print('No configuration reaches development guardrail 99.75%. Inspect misses; no freeze.')
else:
    best=min(feasible,key=lambda r:(r['candidates_mean'],-r['recall'],r['k']))
    dump(ART/'dev/proposed_config.json',dict(k=best['k'],threshold=best['threshold'],minimum=best['minimum'],selection='Smallest mean candidate count among dev configurations with recall >=99.75%',development_metrics=best))
    export('dev',best['k'],threshold=best['threshold'],minimum=best['minimum'])
print(json.dumps(sorted(results,key=lambda r:(-r['recall'],r['candidates_mean']))[:3],indent=2))
