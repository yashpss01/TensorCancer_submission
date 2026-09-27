"""Bounded text-only rescues; every original baseline candidate is preserved."""
import sqlite3,math,collections
from common import *
from normalize import address,bigrams,chargrams,dice
from evaluate import choose as oldchoose,quote
BASE_CFG=load(OLD/'frozen_config.json')
def baseline_set(base):
    return oldchoose(base,**{k:BASE_CFG[k] for k in ['k','threshold','minimum','address_rescue','phonetic_rescue','rescue_empty']})

class Improved:
    def __init__(self,split):
        self.source=sqlite3.connect(f'file:{OLD / "index.sqlite" if split=="dev" else ART / "fresh/index.sqlite"}?mode=ro',uri=True)
        self.db=sqlite3.connect(f'file:{ART/split/"extra.sqlite"}?mode=ro',uri=True)
        self.n=load(OLD/'index_meta.json' if split=='dev' else ART/'fresh/index_meta.json')['target_count']; self.cache={}
    def df(self,col,t):
        if (col,t) not in self.cache: self.cache[col,t]=next((x[0] for x in self.db.execute('SELECT doc FROM vocab WHERE term=? AND col=?',(t,col))),0)
        return self.cache[col,t]
    def rare(self,col,words,k): return sorted(set(words),key=lambda t:(self.df(col,t) or self.n,t))[:k]
    def search(self,expr,k):
        return [x[0] for x in self.db.execute('SELECT rowid FROM extra WHERE extra MATCH ? ORDER BY rank LIMIT ?',(expr,k))] if expr else []
    def records(self,rids):
        return {r[0]:r[1:] for r in self.source.execute('SELECT rowid,id,name,address,country FROM records WHERE rowid IN ('+','.join('?'*len(rids))+')',rids)} if rids else {}
    def addrscore(self,a,b):
        x,y=set(a),set(b)
        if not x or not y: return 0.
        w=lambda t:math.log(1+self.n/(1+self.df('addr',t)))
        inter=x&y; sx=sum(w(t) for t in x); sy=sum(w(t) for t in y)
        s=sum(w(t) for t in inter)/math.sqrt(sx*sy)
        if len(inter)>=3 and any(t.isdigit() for t in inter): s=max(s,.85*sum(w(t) for t in inter)/min(sx,sy))
        return s
    def name_score(self,a,b): return max(dice(chargrams(a),chargrams(b)),.95*dice(bigrams(a),bigrams(b)))
    def retrieve(self,q,base):
        qa=address(q['business_address']); qn=q['business_name']; qb=bigrams(qn)
        expr=lambda col,ts:' OR '.join(col+':'+quote(t) for t in ts)
        ra=self.rare('addr',qa,12); rb=self.rare('n2',qb,6)
        pairs=['(n2:'+quote(a)+' AND n2:'+quote(b)+')' for i,a in enumerate(rb) for b in rb[i+1:]]
        routes={'address2':self.search(expr('addr',ra),40)}
        routes['script2']=self.search('('+' OR '.join(pairs)+') AND ('+expr('addr',ra)+')',40) if pairs and ra else []
        # Only strongly linked, text-distinct candidates can act as one-hop anchors.
        ids=base['ranked'][:12]
        anchors=[]
        if ids:
            rec={r[0]:r[1:] for r in self.source.execute('SELECT id,name,address,country FROM records WHERE id IN ('+','.join('?'*len(ids))+')',ids)}
            seen=set()
            for mid,score in zip(base['ranked'][:12],base['scores'][:12]):
                name,addr,country=rec[mid]; sig=(' '.join(tokens(name)),' '.join(address(addr)))
                if sig in seen: continue
                ns=self.name_score(qn,name); ads=self.addrscore(qa,address(addr))
                if score>=.85 and ads>=.65 and (ns>=.45 or ads>=.9):
                    anchors.append((name,addr)); seen.add(sig)
                if len(anchors)==3: break
        for i,(name,addr) in enumerate(anchors):
            rn=self.rare('native',tokens(name),3)
            # Two rare native-name words + original address context avoid broad alias expansion.
            if len(rn)>=2 and ra:
                ne=' AND '.join('native:'+quote(t) for t in rn[:2])
                routes[f'anchor{i}']=self.search('('+ne+') AND ('+expr('addr',ra)+')',20)
        rids=sorted(set().union(*map(set,routes.values()))); records=self.records(rids)
        # Include retrieved-but-filtered baseline records as possible rescues.
        more=[m for m in base['ranked'] if m not in set(baseline_set(base))]
        if more:
            for r in self.source.execute('SELECT rowid,id,name,address,country FROM records WHERE id IN ('+','.join('?'*len(more))+')',more): records[r[0]]=r[1:]
        scored=[]; existing=set(baseline_set(base))
        for rid,(mid,name,addr,country) in records.items():
            if mid in existing: continue
            sn=self.name_score(qn,name); sa=self.addrscore(qa,address(addr))
            score=max(.95*sa,.6*sn+.4*sa)+.15*min(sn,sa)
            for an,aa in anchors:
                ns=self.name_score(an,name); ads=self.addrscore(address(aa),address(addr))
                score=max(score,.93*(.65*ns+.35*ads)+.08*sa)
            if not addr.strip(): score=max(score,.85*sn)
            scored.append((score,mid))
        scored.sort(key=lambda z:(-z[0],z[1]))
        return dict(id=q['entity_id'],country=q['country'],baseline=baseline_set(base),ranked=[m for s,m in scored],scores=[s for s,m in scored],raw_extra=len(rids),anchors=len(anchors))

def select(r,threshold=.65,cap=12):
    return sorted(set(r['baseline'])|{m for m,s in zip(r['ranked'][:cap],r['scores'][:cap]) if s>=threshold})
