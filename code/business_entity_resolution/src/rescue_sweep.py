"""Development-only investigation of selective route rescue, no new retrieval."""
import json,sqlite3,sys,itertools,time
from blocking import ART,fields,dump
from phonetic import grams
from evaluate import measure, export
start=time.monotonic(); qs=json.loads((ART/'dev/rankings.json').read_text()); truth=json.loads((ART/'dev_truth.json').read_text()); raw={q['entity_id']:q for q in json.loads((ART/'queries.json').read_text()) if q['split']=='dev'}
db=sqlite3.connect(ART/'index.sqlite'); blanks={r[0]:r[1] for r in db.execute("SELECT id,name FROM records WHERE trim(address)='' ")}; bg={i:(set(fields({'business_name':n,'business_address':''})[2].split()),set(grams(n))) for i,n in blanks.items()}
def dice(a,b): return 2*len(a&b)/(len(a)+len(b)) if a and b else 0
for q in qs:
    qg=set(fields(raw[q['id']])[2].split()); pg=set(grams(raw[q['id']]['business_name']))
    eligible=set(q['routes']['name'][:20]+q['routes']['char'][:20]+q['routes']['phonetic'][:20])
    q['empty_rescue']=[m for m in q['ranked'] if m in blanks and m in eligible and max(dice(qg,bg[m][0]),.9*dice(pg,bg[m][1]))>=.5]
dump(ART/'dev/rescue_rankings.json',qs)
res=[]
for k,threshold,addr,ph in itertools.product([20,30,40,60,80],[.4,.45,.5],[0,4,8,12],[0,4,8,12]):
    rows=[]
    for q in qs:
        ids=set(m for m,s in zip(q['ranked'][:k],q['scores']) if s>=threshold)|set(q['empty_rescue'])|set(q['routes']['address'][:addr])|set(q['routes']['phonetic_joint'][:ph])
        rows.append({'id':q['id'],'ranked':list(ids)})
    met=measure(rows,truth,10000,json.loads((ART/'index_meta.json').read_text())['target_count']); res.append(dict(k=k,threshold=threshold,address_rescue=addr,phonetic_rescue=ph,**met))
dump(ART/'dev/rescue_sweep.json',res)
feasible=[r for r in res if r['recall']>=.9975]
print('seconds',time.monotonic()-start)
print(json.dumps(sorted(feasible,key=lambda r:r['candidates_mean'])[:4],indent=2))

# Explicit development-stage budget decision, made before opening holdout:
# accept a 99.70% dev guardrail (still above requested 99.5%) to halve candidate cost.
feasible=[r for r in res if r['recall']>=.997]
best=min(feasible,key=lambda r:(r['candidates_mean'],-r['recall']))
config={k:best[k] for k in ['k','threshold','address_rescue','phonetic_rescue']}
config.update(minimum=0,rescue_empty=True,selection='Smallest mean among selective rescue configurations with dev recall >=99.70%; amended from 99.75% before holdout to reduce candidate cost',development_metrics=best)
dump(ART/'dev/score_only_config.json',json.loads((ART/'dev/proposed_config.json').read_text()))
dump(ART/'dev/proposed_config.json',config)
dump(ART/'dev/rankings.json',qs)
export('dev',config['k'],threshold=config['threshold'],address_rescue=config['address_rescue'],phonetic_rescue=config['phonetic_rescue'],rescue_empty=True)
