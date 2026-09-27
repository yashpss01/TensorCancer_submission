"""Freeze a development-selected configuration before the single holdout run."""
import hashlib,json,pathlib,time
from blocking import ART,dump
p=ART/'frozen_config.json'
if p.exists() or (ART/'HOLDOUT_OPENED.json').exists(): raise RuntimeError('Already frozen/opened; refusing to overwrite')
cfg=json.loads((ART/'dev/proposed_config.json').read_text()); src=pathlib.Path(__file__).parent
cfg['code_sha256']={n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in ['blocking.py','evaluate.py','phonetic.py']}
cfg['queries_sha256']=hashlib.sha256((ART/'queries.json').read_bytes()).hexdigest()
cfg['frozen_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
dump(p,cfg); print(p)
