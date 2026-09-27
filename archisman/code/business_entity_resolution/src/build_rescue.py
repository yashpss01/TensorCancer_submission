"""Create phonetic name rescue index from the bounded pool only."""
import sqlite3,time,argparse
from blocking import ART
from phonetic import grams
p=argparse.ArgumentParser(); p.add_argument('--rebuild',action='store_true'); args=p.parse_args()
start=time.monotonic(); db=sqlite3.connect(ART/'index.sqlite'); db.execute('PRAGMA busy_timeout=60000')
if args.rebuild: db.executescript('DROP TABLE IF EXISTS phonetic_vocab; DROP TABLE IF EXISTS phonetic_search;')
db.execute('CREATE VIRTUAL TABLE phonetic_search USING fts5(grams,content="",detail=full)'); batch=[]
for rid,name in db.execute('SELECT rowid,name FROM records'):
    batch.append((rid,' '.join(grams(name))))
    if len(batch)==10000: db.executemany('INSERT INTO phonetic_search(rowid,grams) VALUES(?,?)',batch); batch=[]
if batch: db.executemany('INSERT INTO phonetic_search(rowid,grams) VALUES(?,?)',batch)
db.execute('CREATE VIRTUAL TABLE phonetic_vocab USING fts5vocab(phonetic_search,row)'); db.commit(); db.close(); print('Rescue index seconds:',time.monotonic()-start)
