"""Disk-backed, label-blind candidate retrieval; stdlib only."""
from phonetic import grams as pgrams
import argparse, collections, csv, hashlib, heapq, json, math, os, pathlib, re, resource, sqlite3, sys, time, unicodedata
ROOT = pathlib.Path(__file__).resolve().parents[3]
ART = ROOT / 'artifacts/blocking'
SEED = 'blocking-first10k-v1-20260926'
FTS_SQL = 'CREATE VIRTUAL TABLE search USING fts5(name,address,grams,phonetic,content="", tokenize="unicode61", detail=full);'
ALIASES = dict(zip('corporation incorporated limited private company road street avenue boulevard drive lane apartment suite floor'.split(), 'corp inc ltd pvt co rd st ave blvd dr ln apt ste fl'.split()))

def tokens(s):
    s = ''.join(c for c in unicodedata.normalize('NFKD', s.casefold()) if not unicodedata.combining(c))
    return [ALIASES.get(t,t) for t in re.findall(r'[^\W_]+',s.replace('&',' and '))]

def fields(row):
    n, a = tokens(row['business_name']), tokens(row['business_address'])
    grams = sorted({w[i:i+3] for w in n for i in range(len(w)-2)})
    return ' '.join(n), ' '.join(a), ' '.join(grams)

def rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        yield from csv.DictReader(f, delimiter='\t')

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')

def prepare():
    ART.mkdir(parents=True,exist_ok=True)
    if (ART/'manifest.json').exists(): raise RuntimeError('Manifest exists; refusing to change frozen split')
    chosen=[]; countries=collections.Counter()
    for r in rows(ROOT/'dataset/train/train_source1.tsv'):
        chosen.append(r); countries[r['country']]+=1
        if len(chosen)==10000: break
    total=len(chosen)
    chosen.sort(key=lambda r:hashlib.sha256((SEED+r['entity_id']).encode()).hexdigest())
    for i,r in enumerate(chosen): r['split']='dev' if i<7000 else 'holdout'
    selected={r['entity_id']:r for r in chosen}
    labels={}
    for r in rows(ROOT/'dataset/train/train_ground_truth.tsv'):
        if r['source1_entity_id'] in selected:
            labels[r['source1_entity_id']]=r['matched_entity_ids'].split(',') if r['matched_entity_ids'] else []
    assert len(labels)==len(chosen)
    owner={}
    for s, mids in labels.items():
        for m in mids:
            assert m not in owner, 'Shared target across sampled groups'
            owner[m]=s
    for r in rows(ROOT/'dataset/train/train_ground_truth.tsv'):
        for m in r['matched_entity_ids'].split(','):
            if m in owner: assert owner[m]==r['source1_entity_id'], 'Sampled group overlaps another reference'
    dump(ART/'queries.json',chosen)
    for split in ['dev','holdout']:
        dump(ART/f'{split}_truth.json',{s:labels[s] for s,r in selected.items() if r['split']==split})
    dump(ART/'manifest.json',dict(seed=SEED,selection='First 10000 S1 file rows; SHA256(seed + S1 ID) ordering: 7000 dev, 3000 holdout',s1_count=total,countries=countries,group_audit='All sampled target labels have exactly one S1 owner across full truth',target_policy='All labeled targets of first 10000 S1 plus SHA256(distractor-v1 + target ID) first 32-bit integer < floor(0.02 * 2**32); no country filtering; reduced-pool optimistic benchmark',holdout_policy='One final evaluation after configuration freeze; further tuning requires a new independent benchmark',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('Prepared immutable 7000 dev / 3000 holdout queries',flush=True)

def build():
    path=ART/'index.sqlite'
    if path.exists(): raise RuntimeError('Index exists; refusing overwrite')
    start=time.monotonic(); db=sqlite3.connect(path)
    db.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-196608; CREATE TABLE records(id TEXT PRIMARY KEY, name TEXT, address TEXT, country TEXT); ' + FTS_SQL)
    required=set()
    for split in ['dev','holdout']:
        for v in json.loads((ART/f'{split}_truth.json').read_text()).values(): required.update(v)
    counts=collections.Counter(); hashes={}; n=0; scanned=0; batch=[]; fbatch=[]
    for source in [2,3]:
        p=ROOT/f'dataset/train/train_source{source}.tsv'
        hashes[p.name]={'bytes':p.stat().st_size}
        for r in rows(p):
            scanned+=1
            if r['entity_id'] not in required and int.from_bytes(hashlib.sha256(('distractor-v1'+r['entity_id']).encode()).digest()[:4],'big')>=int(.02*2**32): continue
            n+=1; counts[r['country']]+=1
            batch.append((n,r['entity_id'],r['business_name'],r['business_address'],r['country']))
            fbatch.append((n,*fields(r),' '.join(pgrams(r['business_name']))))
            if len(batch)==10000:
                db.executemany('INSERT INTO records(rowid,id,name,address,country) VALUES(?,?,?,?,?)',batch)
                db.executemany('INSERT INTO search(rowid,name,address,grams,phonetic) VALUES(?,?,?,?,?)',fbatch)
                db.commit(); batch=[]; fbatch=[]
            if n%250000==0: print(f'Indexed {n:,} targets, {time.monotonic()-start:.0f}s',flush=True)
    if batch:
        db.executemany('INSERT INTO records(rowid,id,name,address,country) VALUES(?,?,?,?,?)',batch)
        db.executemany('INSERT INTO search(rowid,name,address,grams,phonetic) VALUES(?,?,?,?,?)',fbatch)
    db.execute('CREATE VIRTUAL TABLE vocab USING fts5vocab(search, col)'); db.commit()
    dump(ART/'index_meta.json',dict(target_count=n,scanned_targets=scanned,required_targets=len(required),countries=counts,input_files=hashes,build_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,python=sys.version,sqlite=sqlite3.sqlite_version))
    db.close()
    meta_path=ART/'index_meta.json'
    meta=json.loads(meta_path.read_text())
    with open(path,'rb') as f: meta['index_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    dump(meta_path,meta)
    print('INDEX COMPLETE',flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['prepare','build']); a=p.parse_args()
    globals()[a.command]()
if __name__=='__main__': main()
