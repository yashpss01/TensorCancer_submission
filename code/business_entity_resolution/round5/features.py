"""Open-set, text-only pair features for final matching."""
import functools
import math
import re

from common import tokens
from normalize import address, bigrams, chargrams, pkey

LEGAL = {'pvt','ltd','co','corp','inc','llc','plc','limited','private','company','incorporated'}

FEATURE_NAMES = [
    'country_equal','name_exact','name_core_exact','name_first_equal',
    'name_token_count_shared','name_token_dice','name_token_containment','name_token_idf_cosine','name_token_idf_containment',
    'name_core_count_shared','name_core_dice','name_core_containment','name_core_idf_cosine',
    'name_char3_dice','name_core_char3_dice','name_key_bigram_dice','name_key_exact',
    'name_length_ratio','name_core_length_ratio','name_token_count_ratio','name_has_unicode_mismatch',
    'address_exact','address_token_count_shared','address_token_dice','address_token_containment',
    'address_token_idf_cosine','address_token_idf_containment','address_char3_dice',
    'address_number_count_shared','address_number_containment','address_number_conflict','address_postal_equal','address_postal_conflict',
    'address_length_ratio','address_missing_either','address_missing_both',
    'name_address_joint','name_or_address_max','name_address_min',
]


def ratio(a,b):
    return min(a,b)/max(a,b) if max(a,b) else 1.


def dice(a,b):
    return 2*len(a&b)/(len(a)+len(b)) if a and b else 0.


def containment(a,b):
    return len(a&b)/min(len(a),len(b)) if a and b else 0.


def weighted(a,b,weights,default):
    if not a or not b: return 0.,0.
    inter=a&b
    wi=sum(weights.get(x,default) for x in inter)
    wa=sum(weights.get(x,default) for x in a)
    wb=sum(weights.get(x,default) for x in b)
    return wi/math.sqrt(wa*wb),wi/min(wa,wb)


@functools.lru_cache(maxsize=120000)
def represent(name,addr,country):
    name_tokens=tokens(name)
    core=[x for x in name_tokens if x not in LEGAL]
    addr_tokens=address(addr)
    name_norm=' '.join(name_tokens)
    core_norm=' '.join(core)
    addr_norm=' '.join(addr_tokens)
    numbers=frozenset(x for x in addr_tokens if x.isdigit())
    postals=frozenset(x for x in numbers if len(x) in (5,6))
    return dict(country=country,name=name_norm,core=core_norm,first=core[0] if core else '',nt=frozenset(name_tokens),nc=frozenset(core),ng=chargrams(name),cg=chargrams(core_norm),nk=pkey(name),nb=bigrams(name),addr=addr_norm,at=frozenset(addr_tokens),ag=chargrams(addr_norm),nums=numbers,postals=postals,nlen=len(name_norm),clen=len(core_norm),alen=len(addr_norm),unicode_name=any(ord(c)>127 for c in name),missing_addr=not addr.strip())


def features(q,t,name_weights,address_weights,default_weight):
    ns,ni=weighted(q['nt'],t['nt'],name_weights,default_weight)
    cs,_=weighted(q['nc'],t['nc'],name_weights,default_weight)
    ads,adi=weighted(q['at'],t['at'],address_weights,default_weight)
    nd=dice(q['nc'],t['nc']); ad=dice(q['at'],t['at'])
    num_shared=len(q['nums']&t['nums'])
    postal_shared=bool(q['postals']&t['postals'])
    return [
        float(q['country']==t['country']),float(bool(q['name']) and q['name']==t['name']),float(bool(q['core']) and q['core']==t['core']),float(bool(q['first']) and q['first']==t['first']),
        len(q['nt']&t['nt']),dice(q['nt'],t['nt']),containment(q['nt'],t['nt']),ns,ni,
        len(q['nc']&t['nc']),nd,containment(q['nc'],t['nc']),cs,
        dice(q['ng'],t['ng']),dice(q['cg'],t['cg']),dice(q['nb'],t['nb']),float(bool(q['nk']) and q['nk']==t['nk']),
        ratio(q['nlen'],t['nlen']),ratio(q['clen'],t['clen']),ratio(len(q['nt']),len(t['nt'])),float(q['unicode_name']!=t['unicode_name']),
        float(bool(q['addr']) and q['addr']==t['addr']),len(q['at']&t['at']),ad,containment(q['at'],t['at']),
        ads,adi,dice(q['ag'],t['ag']),
        num_shared,containment(q['nums'],t['nums']),float(bool(q['nums'] and t['nums']) and num_shared==0),float(postal_shared),float(bool(q['postals'] and t['postals']) and not postal_shared),
        ratio(q['alen'],t['alen']),float(q['missing_addr'] or t['missing_addr']),float(q['missing_addr'] and t['missing_addr']),
        nd*ad,max(nd,ad),min(nd,ad),
    ]


assert len(FEATURE_NAMES)==len(features(represent('a','b','US'),represent('a','b','US'),{}, {},1.))
