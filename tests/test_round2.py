import unittest,sys,pathlib,hashlib,json
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/business_entity_resolution/round2'))
from normalize import address,pkey,bigrams
from improve import select
class Round2Tests(unittest.TestCase):
    def test_baseline_never_removed(self):
        r={'baseline':['a','b'],'ranked':['b','c','d'],'scores':[.9,.8,.5]}
        for cap in [0,1,2,10]:
            for threshold in [0.,.7,1.1]: self.assertTrue({'a','b'}<=set(select(r,threshold,cap)))
        self.assertEqual(select(r,.7,2),['a','b','c'])
    def test_ordinal_state_and_zero_variants(self):
        self.assertEqual(address('0315 First Street, Texas'),address('315 1st St, TX'))
        self.assertEqual(address('Maharashtra, 00201'),['mh','201'])
        self.assertIn('ecole',address('École 12 Rue Victor Hugo France'))
    def test_cross_script_suffix(self):
        self.assertGreater(len(bigrams('Green Solutions')&bigrams('ग्रीन सॉल्यूशंस')),2)
    def test_original_code_frozen(self):
        cfg=json.loads((ROOT/'artifacts/blocking/frozen_config.json').read_text())
        for n,d in cfg['code_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/'code/business_entity_resolution/src'/n).read_bytes()).hexdigest(),d)
if __name__=='__main__': unittest.main()
