"""Text-only normalization. Originals stay in the source records."""
import re
from functools import lru_cache
from blocking import tokens
from phonetic import romanize,key as oldkey
# Conventional structural address abbreviations, not business enrichment.
REGIONS={'andhra pradesh':'ap','arunachal pradesh':'ar','assam':'as','bihar':'br','chhattisgarh':'cg','goa':'ga','gujarat':'gj','haryana':'hr','himachal pradesh':'hp','jharkhand':'jh','karnataka':'ka','kerala':'kl','keralam':'kl','madhya pradesh':'mp','maharashtra':'mh','manipur':'mn','meghalaya':'ml','mizoram':'mz','nagaland':'nl','odisha':'od','orissa':'od','punjab':'pb','rajasthan':'rj','sikkim':'sk','tamil nadu':'tn','telangana':'tg','tripura':'tr','uttar pradesh':'up','uttarakhand':'uk','west bengal':'wb','new delhi':'delhi','alabama':'al','alaska':'ak','arizona':'az','arkansas':'ar','california':'ca','colorado':'co','connecticut':'ct','delaware':'de','florida':'fl','georgia':'ga','hawaii':'hi','idaho':'id','illinois':'il','indiana':'in','iowa':'ia','kansas':'ks','kentucky':'ky','louisiana':'la','maine':'me','maryland':'md','massachusetts':'ma','michigan':'mi','minnesota':'mn','mississippi':'ms','missouri':'mo','montana':'mt','nebraska':'ne','nevada':'nv','new hampshire':'nh','new jersey':'nj','new mexico':'nm','new york':'ny','north carolina':'nc','north dakota':'nd','ohio':'oh','oklahoma':'ok','oregon':'or','pennsylvania':'pa','rhode island':'ri','south carolina':'sc','south dakota':'sd','tennessee':'tn','texas':'tx','utah':'ut','vermont':'vt','virginia':'va','washington':'wa','west virginia':'wv','wisconsin':'wi','wyoming':'wy'}
ORD={'first':'1','second':'2','third':'3','fourth':'4','fifth':'5','sixth':'6','seventh':'7','eighth':'8','ninth':'9','tenth':'10','eleventh':'11','twelfth':'12','thirteenth':'13','fourteenth':'14','fifteenth':'15','sixteenth':'16','seventeenth':'17','eighteenth':'18','nineteenth':'19','twentieth':'20'}
REPLACE=re.compile(r'\b('+'|'.join(re.escape(x) for x in sorted(REGIONS,key=len,reverse=True))+r')\b')
@lru_cache(maxsize=16000)
def address(text):
    s=' '.join(tokens(text)); s=REPLACE.sub(lambda m:REGIONS[m[0]],s)
    s=re.sub(r'\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety) (first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)\b',lambda m:str({'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90}[m[1]]+int(ORD[m[2]])),s)
    out=[]
    for t in s.split():
        t=ORD.get(t,t)
        if re.fullmatch(r'\d+(st|nd|rd|th)?',t): t=str(int(re.match(r'\d+',t)[0]))
        t={'saint':'st','mount':'mt','avenue':'ave','boulevard':'blvd','route':'rte','highway':'hwy','hno':'house','rd':'rd','null':''}.get(t,t)
        if t: out.append(t)
    return out

@lru_cache(maxsize=16000)
def pkey(text):
    s=romanize(text).replace('tion','shan').replace('sion','shan')
    return re.sub(r'(.)\1+',r'\1',oldkey(s).translate(str.maketrans({'m':'n','d':'t','b':'p'})))
@lru_cache(maxsize=16000)
def bigrams(text):
    s=pkey(text); return {s[i:i+2] for i in range(len(s)-1)}
@lru_cache(maxsize=16000)
def chargrams(text):
    s=' '.join(tokens(text)); return {s[i:i+3] for i in range(len(s)-2)}
def dice(x,y):
    x,y=set(x),set(y); return 2*len(x&y)/(len(x)+len(y)) if x and y else 0.
