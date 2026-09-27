import sys, pathlib, unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'code/business_entity_resolution/src'))
from blocking import fields, tokens, FTS_SQL
from evaluate import measure, choose

class MetricTests(unittest.TestCase):
    def test_counts_and_singleton_oracle(self):
        q=[dict(id='a',ranked=['x','z']),dict(id='b',ranked=['y']),dict(id='c',ranked=[])]
        t={'a':['x','w'],'b':[],'c':['v']}
        m=measure(q,t,2,10)
        self.assertEqual((m['TP'],m['FN'],m['FP'],m['TN']),(1,2,2,25))
        self.assertAlmostEqual(m['recall'],1/3)
        self.assertAlmostEqual(m['precision'],1/3)
        self.assertAlmostEqual(m['reduction_ratio'],.9)
        self.assertAlmostEqual(m['oracle_macro_f05'],(5/6+1+0)/3)
    def test_fts_rank_uses_term_evidence(self):
        import sqlite3
        db=sqlite3.connect(':memory:'); db.execute(FTS_SQL)
        db.executemany('INSERT INTO search(rowid,name,address,grams) VALUES(?,?,?,?)',[(1,'cafe other','',''),(2,'fresh cafe','',''),(3,'fresh unrelated','','')])
        scores=db.execute('SELECT rowid,rank FROM search WHERE search MATCH ? ORDER BY rank',('name:fresh OR name:cafe',)).fetchall()
        self.assertEqual(scores[0][0],2)
        self.assertLess(scores[0][1],0)
        self.assertLess(scores[0][1],scores[-1][1])
    def test_cutoff_is_exact_final_set(self):
        q=dict(ranked=['a','b','c'],scores=[.9,.5,.2])
        self.assertEqual(choose(q,2,.6),['a'])
        self.assertEqual(choose(q,1,0.),['a'])
        self.assertEqual(choose(q,3,.6,2),['a','b'])
    def test_selective_rescues_are_in_final_set(self):
        q=dict(id='x',ranked=['a','b','c','d'],scores=[.9,.4,.3,.2],routes={'address':['c'],'phonetic_joint':[]},empty_rescue=['d'])
        self.assertEqual(choose(q,1,.5,address_rescue=1,rescue_empty=True),['a','c','d'])
        m=measure([q],{'x':['d']},1,10,threshold=.5,address_rescue=1,rescue_empty=True)
        self.assertEqual((m['TP'],m['FP'],m['FN'],m['TN']),(1,2,0,7))
    def test_unicode_phonetic_derivation(self):
        from phonetic import key
        self.assertEqual(key('Sai Infotech Private Limited'),key('साईं इंफोटेक प्राइवेट लिमिटेड'))
        self.assertEqual(key('Café'),key('Cafe'))
    def test_perfect_oracle(self):
        q=[dict(id='a',ranked=['x','false']),dict(id='b',ranked=['false'])]
        m=measure(q,{'a':['x'],'b':[]},2,10)
        self.assertEqual(m['oracle_macro_f05'],1.)
        self.assertEqual(m['FN'],0)
    def test_no_id_country_features_and_original_unchanged(self):
        r=dict(entity_id='S1-1',business_name='Café Private Limited',business_address='12 ROAD',country='France')
        original=dict(r); f=fields(r)
        r['entity_id']='S3-999'; r['country']='Unseen'
        self.assertEqual(f,fields(r))
        self.assertEqual(original['business_name'],r['business_name'])
        self.assertEqual(tokens('Café Private Limited'),['cafe','pvt','ltd'])
if __name__=='__main__': unittest.main()
