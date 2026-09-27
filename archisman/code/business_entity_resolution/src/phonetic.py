"""Algorithmic Unicode-name Indic transliteration; no external vocabulary or lookup."""
import re, unicodedata
SCRIPTS=('DEVANAGARI','BENGALI','GUJARATI','GURMUKHI','ORIYA','TAMIL','TELUGU','KANNADA','MALAYALAM')
LEGAL={'private','limited','pvt','ltd','llc','inc','incorporated','corporation','corp','company','co'}
def romanize(text):
    out=[]
    for c in unicodedata.normalize('NFC',text.casefold()):
        name=unicodedata.name(c,'')
        if name.startswith(SCRIPTS):
            if 'VIRAMA' in name:
                if out and out[-1].endswith('a'): out[-1]=out[-1][:-1]
            elif 'VOWEL SIGN ' in name:
                if out and out[-1].endswith('a'): out[-1]=out[-1][:-1]
                out.append(name.split('VOWEL SIGN ')[-1].lower().replace('vocalic ','').replace('candra ',''))
            elif 'LETTER ' in name:
                out.append(name.split('LETTER ')[-1].lower().replace('vocalic ','').replace('candra ',''))
            elif 'ANUSVARA' in name:
                out.append('n')
            elif 'DIGIT ' in name:
                try: out.append(str(unicodedata.digit(c)))
                except ValueError: pass
            # Nasalization/nukta marks are ignored in this intentionally coarse key.
        else:
            out.append(''.join(x for x in unicodedata.normalize('NFKD',c) if not unicodedata.combining(x)))
    return ''.join(out)
def key(text):
    ws=re.findall('[a-z0-9]+',romanize(text)); keys=[]
    for w in ws:
        if w in LEGAL or w=='li': continue
        w=re.sub(r'c(?=[ei])','s',w).replace('ph','f').replace('sh','s').replace('ch','c')
        w=w.translate(str.maketrans({'q':'k','c':'k','w':'v','z':'s','j':'s','y':'i','x':'ks'}))
        w=re.sub('[aeiouh]','',w); w=re.sub(r'(.)\1+',r'\1',w)
        if w and w not in {'prvt','lmtd','lt','pv','pr'}: keys.append(w)
    return re.sub(r'(.)\1+',r'\1',''.join(keys))
def grams(text):
    s=key(text)
    return sorted({s[i:i+3] for i in range(len(s)-2)})
