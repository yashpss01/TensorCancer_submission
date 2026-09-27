"""Rebuild derived FTS fields from the saved bounded pool, without rescanning input data."""
from phonetic import grams as pgrams
import hashlib, json, resource, sqlite3, time
from blocking import ART, fields, dump, FTS_SQL
start=time.monotonic(); db=sqlite3.connect(ART/'index.sqlite')
db.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-131072; DROP TABLE vocab; DROP TABLE search; ' + FTS_SQL)
batch=[]
for rid,name,address in db.execute('SELECT rowid,name,address FROM records'):
    batch.append((rid,*fields(dict(business_name=name,business_address=address)),' '.join(pgrams(name))))
    if len(batch)==10000:
        db.executemany('INSERT INTO search(rowid,name,address,grams,phonetic) VALUES(?,?,?,?,?)',batch); batch=[]
if batch: db.executemany('INSERT INTO search(rowid,name,address,grams,phonetic) VALUES(?,?,?,?,?)',batch)
db.execute('CREATE VIRTUAL TABLE vocab USING fts5vocab(search,col)'); db.commit(); db.close()
p=ART/'index_meta.json'; meta=json.loads(p.read_text()); meta['repair_seconds']=time.monotonic()-start; meta['repair_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
with open(ART/'index.sqlite','rb') as f: meta['index_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
dump(p,meta)
print('Rebuilt full-position index',meta['repair_seconds'])
