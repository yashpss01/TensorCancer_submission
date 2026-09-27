import pathlib,sys,json
SRC=pathlib.Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
from blocking import ROOT,rows,dump,fields,tokens,FTS_SQL
OLD=ROOT/'artifacts/blocking'; ART=ROOT/'artifacts/blocking_round2'
def load(p): return json.loads(p.read_text())
