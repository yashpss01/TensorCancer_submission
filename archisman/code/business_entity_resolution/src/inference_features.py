"""Inference-only feature functions matching the frozen round-5 models."""
import difflib
import functools

import numpy as np

from features import dice,represent,weighted

EXTRA_NAMES=['core_sequence_ratio','full_name_sequence_ratio','address_sequence_ratio','core_sorted_sequence_ratio','name_key_sequence_ratio','core_longest_match','address_longest_match','core_soft_overlap','core_soft_containment','core_char_idf_cosine','address_char_idf_cosine','target_address_semantically_missing','target_domain_name','query_domain_name','first_address_number_equal','long_address_number_conflict','address_number_query_contained','address_number_target_contained','name_token_count_difference','address_token_count_difference']
GROUP_NAMES=['pair_probability','group_best_probability','group_second_probability','group_other_best_probability','group_high_count','group_moderate_count','probability_rank_fraction','probability_gap_to_best','anchor_count','anchor_name_core_dice','anchor_name_char3_dice','anchor_name_key_dice','anchor_address_dice','anchor_address_idf_cosine','anchor_address_number_containment','anchor_name_address_joint','anchor_core_exact','anchor_address_exact','anchor_support_count','name_core_sequence','name_full_sequence','name_key_sequence','address_sequence','address_first_number_equal','address_long_number_conflict','address_query_numbers_contained','address_target_numbers_contained']


@functools.lru_cache(maxsize=100_000)
def seq(a,b):
    if not a or not b:return 0.,0.
    matcher=difflib.SequenceMatcher(None,a,b,autojunk=False)
    return matcher.ratio(),matcher.find_longest_match(0,len(a),0,len(b)).size/min(len(a),len(b))


@functools.lru_cache(maxsize=100_000)
def soft(a,b):
    if not a or not b:return 0.,0.
    x=a.split();y=b.split()
    total=0.
    for word in x:
        best=max((difflib.SequenceMatcher(None,word,other,autojunk=False).ratio() for other in y),default=0.)
        if best>=.55:total+=best
    return total/max(len(x),len(y)),total/min(len(x),len(y))


def extra_vector(q,t,nw,aw,default):
    core,long_core=seq(q['core'],t['core'])
    name,_=seq(q['name'],t['name'])
    addr,long_addr=seq(q['addr'],t['addr'])
    sorted_core,_=seq(' '.join(sorted(q['core'].split())),' '.join(sorted(t['core'].split())))
    key,_=seq(q['nk'],t['nk'])
    ss,sc=soft(q['core'],t['core'])
    nc,_=weighted(q['cg'],t['cg'],nw,default)
    ac,_=weighted(q['ag'],t['ag'],aw,default)
    qfirst=next((x for x in q['addr'].split() if x.isdigit()),'')
    tfirst=next((x for x in t['addr'].split() if x.isdigit()),'')
    qlong={x for x in q['nums'] if len(x)>=3}
    tlong={x for x in t['nums'] if len(x)>=3}
    missing=not t['addr'] or t['addr'] in {'n a','na','none','unknown','not available','nil'}
    return [core,name,addr,sorted_core,key,long_core,long_addr,ss,sc,nc,ac,float(missing),float('com' in t['nt'] or 'www' in t['nt']),float('com' in q['nt'] or 'www' in q['nt']),float(bool(qfirst and tfirst) and qfirst==tfirst),float(bool(qlong and tlong) and not (qlong&tlong)),float(bool(q['nums']) and q['nums']<=t['nums']),float(bool(t['nums']) and t['nums']<=q['nums']),abs(len(q['nt'])-len(t['nt'])),abs(len(q['at'])-len(t['at']))]


def group_matrix(query,reps,prob,address_weights,default):
    """Return frozen group features in candidate order (including pair probability)."""
    if not len(prob):return np.empty((0,len(GROUP_NAMES)),dtype='float32')
    ps=np.asarray(prob)
    order=np.argsort(-ps,kind='stable')
    top=order[:3]
    anchors=[int(i) for i in top if ps[i]>=.9]
    if not anchors:anchors=[int(order[0])]
    qrep=represent(query['business_name'],query['business_address'],query['country'])
    best=float(ps[order[0]]);second=float(ps[order[1]]) if len(order)>1 else 0.
    high=int((ps>=.9).sum());moderate=int((ps>=.5).sum())
    ranks=np.empty(len(ps),dtype=np.int32);ranks[order]=np.arange(len(ps))
    result=np.empty((len(ps),len(GROUP_NAMES)),dtype='float32')
    qfirst=next((x for x in qrep['addr'].split() if x.isdigit()),'')
    qlong={x for x in qrep['nums'] if len(x)>=3}
    for j,rep in enumerate(reps):
        other=[a for a in anchors if a!=j]
        values=[]
        for a in other:
            ar=reps[a]
            ad,_=weighted(rep['at'],ar['at'],address_weights,default)
            nc=dice(rep['nc'],ar['nc']);aa=dice(rep['at'],ar['at'])
            values.append((nc,dice(rep['cg'],ar['cg']),dice(rep['nb'],ar['nb']),aa,ad,len(rep['nums']&ar['nums'])/min(len(rep['nums']),len(ar['nums'])) if rep['nums'] and ar['nums'] else 0.,nc*aa,float(bool(rep['core']) and rep['core']==ar['core']),float(bool(rep['addr']) and rep['addr']==ar['addr']),float(ps[a])))
        maxima=[max(v[k] for v in values) if values else 0. for k in range(10)]
        support=sum(v[6]>.4 and v[9]>.9 for v in values)
        other_best=second if j==order[0] else best
        # The same comparisons were computed by extra_vector for this pair.
        # seq() caches their exact SequenceMatcher results within a bounded size.
        sequence=lambda a,b:seq(a,b)[0]
        tfirst=next((x for x in rep['addr'].split() if x.isdigit()),'')
        tlong={x for x in rep['nums'] if len(x)>=3}
        result[j]=[float(ps[j]),best,second,other_best,high,moderate,float(ranks[j])/max(1,len(ps)-1),best-float(ps[j]),len(other),*maxima[:9],support,
            sequence(qrep['core'],rep['core']),sequence(qrep['name'],rep['name']),sequence(qrep['nk'],rep['nk']),sequence(qrep['addr'],rep['addr']),float(bool(qfirst and tfirst) and qfirst==tfirst),float(bool(qlong and tlong) and not (qlong&tlong)),float(bool(qrep['nums']) and qrep['nums']<=rep['nums']),float(bool(rep['nums']) and rep['nums']<=qrep['nums'])]
    return result
