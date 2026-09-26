"""Check exact candidate TSVs against the organizer's validator helper and saved rankings."""
import csv, importlib.util, json, sqlite3
from blocking import ROOT, ART
from evaluate import choose
spec=importlib.util.spec_from_file_location('validator',ROOT/'utils/validate_submission.py'); v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
db=sqlite3.connect(ART/'index.sqlite'); valid={r[0] for r in db.execute('SELECT id FROM records')}
queries=json.loads((ART/'queries.json').read_text()); cfg=json.loads((ART/'frozen_config.json').read_text())
all_truth=set()
for split in ['dev','holdout']:
    truth=json.loads((ART/f'{split}_truth.json').read_text())
    for matches in truth.values(): all_truth.update(matches)
    required={q['entity_id'] for q in queries if q['split']==split}; errors=[]
    mapping=v.validate_id_list_file(str(ART/split/'candidate_pairs.tsv'),v.CANDIDATE_HEADER,'candidate_entity_ids',required,valid,errors)
    assert not errors, errors
    rankings=json.loads((ART/split/'rankings.json').read_text())
    assert mapping=={q['id']:set(choose(q,cfg['k'],cfg.get('threshold',0.),cfg.get('minimum',0),cfg.get('address_rescue',0),cfg.get('phonetic_rescue',0),cfg.get('rescue_empty',False))) for q in rankings}
    print(split,': exact frozen candidate set and format verified')
assert all_truth<=valid, f'{len(all_truth-valid)} missing labeled targets'
print('PASS: all labeled targets exist in pool; exact exported candidate sets verified')

# Convenient combined artifact; original S1 file order, exact frozen sets only.
from itertools import islice
from blocking import rows
combined={}
for split in ['dev','holdout']:
    for row in rows(ART/split/'candidate_pairs.tsv'):
        combined[row['source1_entity_id']]=row['candidate_entity_ids']
with open(ART/'candidate_pairs.tsv','w') as f:
    f.write('source1_entity_id\tcandidate_entity_ids\n')
    for q in islice(rows(ROOT/'dataset/train/train_source1.tsv'),10000):
        f.write(q['entity_id']+'\t'+combined[q['entity_id']]+'\n')
assert len(combined)==10000
print('Combined first-10000 candidate artifact written')
