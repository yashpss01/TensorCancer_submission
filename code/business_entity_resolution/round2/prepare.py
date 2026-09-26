"""Only benchmark construction sees new labels before freeze; no matching evaluation."""
import hashlib,itertools,sqlite3,collections,time,resource
from common import *
from phonetic import grams
(ART/'dev').mkdir(parents=True,exist_ok=True)
(ART/'fresh').mkdir(parents=True,exist_ok=True)
if (ART/'fresh/manifest.json').exists(): raise RuntimeError('Fresh manifest already exists')
start=time.monotonic(); old=load(OLD/'queries.json'); fresh=list(itertools.islice(rows(ROOT/'dataset/train/train_source1.tsv'),10000,20000))
assert len(fresh)==10000
old_ids={q['entity_id'] for q in old}; new_ids={q['entity_id'] for q in fresh}; assert not old_ids&new_ids
selected=old_ids|new_ids
labels={r['source1_entity_id']:(r['matched_entity_ids'].split(',') if r['matched_entity_ids'] else []) for r in rows(ROOT/'dataset/train/train_ground_truth.tsv') if r['source1_entity_id'] in selected}
owner={}
for sid,mids in labels.items():
    for mid in mids:
        assert mid not in owner, 'Overlapping labeled entity groups'
        owner[mid]=sid
for r in rows(ROOT/'dataset/train/train_ground_truth.tsv'):
    for mid in r['matched_entity_ids'].split(','):
        if mid in owner: assert owner[mid]==r['source1_entity_id'], 'Shared ownership outside benchmark'
old_targets={m for s in old_ids for m in labels[s]}; required={m for s in new_ids for m in labels[s]}
assert not old_targets&required
sig=lambda q:(tuple(fields(q)[:2]),q['country'])
exact_overlap=len({sig(q) for q in old}&{sig(q) for q in fresh})
dump(ART/'fresh/queries.json',fresh); dump(ART/'fresh/truth.json',{s:labels[s] for s in new_ids})
dump(ART/'dev/queries.json',old); dump(ART/'dev/truth.json',{s:labels[s] for s in old_ids})
db=sqlite3.connect(ART/'fresh/index.sqlite'); db.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; CREATE TABLE records(id TEXT PRIMARY KEY,name TEXT,address TEXT,country TEXT);'+FTS_SQL)
batch=[]; fb=[]; n=0; scanned=0; countries=collections.Counter()
for src in [2,3]:
    for r in rows(ROOT/f'dataset/train/train_source{src}.tsv'):
        scanned+=1
        if r['entity_id'] not in required and int.from_bytes(hashlib.sha256(('distractor-v1'+r['entity_id']).encode()).digest()[:4],'big')>=int(.02*2**32): continue
        n+=1; countries[r['country']]+=1
        batch.append((n,r['entity_id'],r['business_name'],r['business_address'],r['country']))
        fb.append((n,*fields(r),' '.join(grams(r['business_name']))))
        if len(batch)>=10000:
            db.executemany('INSERT INTO records(rowid,id,name,address,country) VALUES(?,?,?,?,?)',batch); db.executemany('INSERT INTO search(rowid,name,address,grams,phonetic) VALUES(?,?,?,?,?)',fb); db.commit(); batch=[]; fb=[]
if batch:
    db.executemany('INSERT INTO records(rowid,id,name,address,country) VALUES(?,?,?,?,?)',batch); db.executemany('INSERT INTO search(rowid,name,address,grams,phonetic) VALUES(?,?,?,?,?)',fb)
db.execute('CREATE VIRTUAL TABLE vocab USING fts5vocab(search,col)'); db.commit(); db.close()
meta=dict(target_count=n,required_targets=len(required),distractors=n-len(required),scanned_targets=scanned,countries=countries,seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
dump(ART/'fresh/index_meta.json',meta)
dump(ART/'fresh/manifest.json',dict(rows='10001 through 20000 inclusive in original S1 order',reference_count=10000,target_policy='All labeled positives plus deterministic 2% SHA256(distractor-v1 + target ID) sample; no country restriction',shared_s1_ids=0,shared_labeled_targets=0,group_audit='All first-20000 targets have only one S1 owner across full truth',exact_normalized_name_address_country_overlap=exact_overlap,development='First 10000 including previously inspected holdout; no longer untouched',seal='Do not evaluate fresh outcomes until improvement and selection code/configuration are frozen',built_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
print('Fresh benchmark constructed; outcomes sealed',meta,flush=True)
