import sqlite3,time,resource,argparse
from common import *
from normalize import address,bigrams

def build(split):
    out=ART/split/'extra.sqlite'
    if out.exists(): raise RuntimeError('Extra index exists')
    source=OLD/'index.sqlite' if split=='dev' else ART/'fresh/index.sqlite'
    src=sqlite3.connect(f'file:{source}?mode=ro',uri=True); db=sqlite3.connect(out); start=time.monotonic()
    db.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; CREATE VIRTUAL TABLE extra USING fts5(native,addr,n2,content="",detail=full);')
    batch=[]
    for rid,name,addr in src.execute('SELECT rowid,name,address FROM records'):
        batch.append((rid,' '.join(tokens(name)),' '.join(address(addr)),' '.join(sorted(bigrams(name)))))
        if len(batch)==10000: db.executemany('INSERT INTO extra(rowid,native,addr,n2) VALUES(?,?,?,?)',batch); batch=[]
    if batch: db.executemany('INSERT INTO extra(rowid,native,addr,n2) VALUES(?,?,?,?)',batch)
    db.execute('CREATE VIRTUAL TABLE vocab USING fts5vocab(extra,col)'); db.commit(); db.close()
    dump(ART/split/'extra_meta.json',dict(seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,bytes=out.stat().st_size)); print('Extra index complete:',split)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('split',choices=['dev','fresh']); build(p.parse_args().split)
