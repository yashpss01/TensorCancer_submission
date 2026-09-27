"""Text normalisation for business names and addresses.

Everything here is rule based and language agnostic where possible: the same
function is applied to every country (US, India, France and any unseen label).
Country specific dictionaries (state abbreviations, street words) are only
*additional* expansions; they never filter records.
"""
from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

# --------------------------------------------------------------------------- #
# Legal-form / generic tokens (kept as a separate "legal" signal, stripped from
# the "core" name).
# --------------------------------------------------------------------------- #
LEGAL_MAP = {
    # english / india
    "private": "pvt", "pvt": "pvt", "pvtltd": "pvt ltd", "pvt.": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltd.": "ltd", "lted": "ltd", "limted": "ltd",
    "incorporated": "inc", "inc": "inc", "inc.": "inc",
    "corporation": "corp", "corp": "corp", "corp.": "corp",
    "company": "co", "co": "co", "co.": "co", "companies": "co",
    "llc": "llc", "l.l.c": "llc", "l.l.c.": "llc", "pllc": "pllc",
    "llp": "llp", "l.l.p": "llp", "lp": "lp", "l.p.": "lp",
    "plc": "plc", "pc": "pc", "p.c.": "pc", "pa": "pa",
    "gmbh": "gmbh", "ag": "ag", "aktiengesellschaft": "ag",
    "oyj": "oyj", "oy": "oy", "ab": "ab", "sa": "sa", "s.a.": "sa", "nv": "nv",
    "bv": "bv", "srl": "srl", "spa": "spa", "kk": "kk",
    "opc": "opc", "public": "public",
    # french
    "sarl": "sarl", "s.a.r.l": "sarl", "s.a.r.l.": "sarl", "sas": "sas", "s.a.s": "sas",
    "s.a.s.": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci", "snc": "snc",
    "scp": "scp", "sel": "sel", "selarl": "selarl", "sca": "sca", "gie": "gie",
    "societe": "ste", "société": "ste", "ste": "ste", "cie": "cie", "compagnie": "cie",
    "ets": "ets", "etablissements": "ets", "établissements": "ets",
    "groupe": "group", "group": "group",
}
LEGAL_TOKENS = set(LEGAL_MAP.values()) | {"and", "of", "the", "de", "du", "des", "la", "le", "les", "et"}

# Words the noise generator likes to bolt on / drop.  They carry almost no
# identity information, so they get a low weight in the "core" name.
FILLER_TOKENS = {
    "center", "centre", "services", "service", "solutions", "solution", "enterprises",
    "enterprise", "group", "holdings", "holding", "international", "india", "global",
    "industries", "industry", "trading", "traders", "associates", "partners", "dr", "mr",
    "mrs", "smt", "shri", "sri", "sh", "m/s", "ms", "messrs", "the", "of", "and",
}

# --------------------------------------------------------------------------- #
# Address abbreviations (US + India + France).  Keys are lower-case tokens.
# --------------------------------------------------------------------------- #
ADDR_ABBR = {
    # US street types
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "dr": "drive", "drv": "drive", "ct": "court", "crt": "court", "ln": "lane", "la": "lane",
    "blvd": "boulevard", "bld": "boulevard", "hwy": "highway", "pkwy": "parkway",
    "pky": "parkway", "cir": "circle", "pl": "place", "ter": "terrace", "terr": "terrace",
    "trl": "trail", "tr": "trail", "way": "way", "wy": "way", "sq": "square", "plz": "plaza",
    "pt": "point", "mt": "mount", "ste": "suite", "apt": "apartment", "fl": "floor",
    "flr": "floor", "bldg": "building", "rte": "route", "hts": "heights", "hl": "hill",
    "hls": "hills", "xing": "crossing", "expy": "expressway", "fwy": "freeway",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "po": "po", "cty": "county", "twp": "township",
    "ft": "fort", "jct": "junction", "lk": "lake", "vlg": "village", "cdp": "",
    # India
    "nr": "near", "opp": "opposite", "bldg": "building", "flr": "floor", "gr": "ground",
    "sec": "sector", "sect": "sector", "ph": "phase", "colony": "colony", "nagar": "nagar",
    "mkt": "market", "indl": "industrial", "ind": "industrial", "estt": "estate", "est": "estate",
    "hno": "", "h.no": "", "sno": "", "s.no": "", "plot": "plot", "dist": "", "distt": "",
    "tq": "", "taluka": "", "tal": "", "po": "po", "vill": "village", "vil": "village",
    "marg": "marg", "chowk": "chowk", "gali": "gali", "sq": "square", "apts": "apartment",
    "apt": "apartment", "appt": "apartment", "chs": "chs", "soc": "society", "socy": "society",
    "extn": "extension", "ext": "extension", "encl": "enclave", "compl": "complex", "cplx": "complex",
    "blk": "block", "gf": "ground floor", "ff": "first floor", "sf": "second floor",
    "tf": "third floor", "1st": "1", "2nd": "2", "3rd": "3", "4th": "4", "5th": "5", "6th": "6",
    "7th": "7", "8th": "8", "9th": "9", "10th": "10", "ist": "1", "iind": "2", "iiird": "3",
    "iind": "2", "ii": "2", "iii": "3", "iv": "4",
    # France
    "r": "rue", "r.": "rue", "bd": "boulevard", "bvd": "boulevard", "boul": "boulevard",
    "pl": "place", "imp": "impasse", "all": "allee", "allée": "allee", "ch": "chemin", "che": "chemin",
    "chem": "chemin", "rte": "route", "sq": "square", "crs": "cours", "qu": "quai", "fbg": "faubourg",
    "fg": "faubourg", "pass": "passage", "res": "residence", "rés": "residence", "lot": "lotissement",
    "za": "za", "zi": "zi", "zac": "zac", "st": "street", "ste": "suite", "cedex": "",
    "bis": "bis", "ter": "terrace",
}

# US state abbreviations -> full names (and reverse map built below)
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "dc": "district of columbia", "pr": "puerto rico",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachal pradesh", "jk": "jammu and kashmir", "jh": "jharkhand", "ka": "karnataka",
    "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur",
    "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "odisha",
    "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana",
    "ts": "telangana", "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand",
    "ua": "uttarakhand", "wb": "west bengal", "dl": "delhi", "ch": "chandigarh",
    "py": "puducherry", "pd": "puducherry", "an": "andaman and nicobar", "ld": "lakshadweep",
    "dn": "dadra and nagar haveli", "dd": "daman and diu", "nct": "delhi",
}
# Indic-script state names that appear verbatim in addresses.
INDIC_STATE = {
    "महाराष्ट्र": "maharashtra", "मध्य प्रदेश": "madhya pradesh", "उत्तर प्रदेश": "uttar pradesh",
    "दिल्ली": "delhi", "राजस्थान": "rajasthan", "बिहार": "bihar", "हरियाणा": "haryana",
    "गुजरात": "gujarat", "पंजाब": "punjab", "झारखंड": "jharkhand", "छत्तीसगढ़": "chhattisgarh",
    "उत्तराखंड": "uttarakhand", "हिमाचल प्रदेश": "himachal pradesh", "ਪੰਜਾਬ": "punjab",
    "ಕರ್ನಾಟಕ": "karnataka", "தமிழ்நாடு": "tamil nadu", "தமிழ் நாடு": "tamil nadu",
    "కర్ణాటక": "karnataka", "తెలంగాణ": "telangana", "ఆంధ్ర ప్రదేశ్": "andhra pradesh",
    "കേരളം": "kerala", "പശ്ചിമ ബംഗാൾ": "west bengal", "পশ্চিমবঙ্গ": "west bengal",
    "ગુજરાત": "gujarat", "ओडिशा": "odisha", "ଓଡ଼ିଶା": "odisha", "असम": "assam", "অসম": "assam",
    "गोवा": "goa", "চণ্ডীগড়": "chandigarh", "चंडीगढ़": "chandigarh",
}
STATE_ABBR = {}
STATE_ABBR.update({k: v for k, v in US_STATES.items()})
# Indian abbreviations override US ones only where they do not clash badly.
for k, v in IN_STATES.items():
    STATE_ABBR.setdefault(k, v)
FULL_TO_ABBR = {v: k for k, v in list(US_STATES.items()) + list(IN_STATES.items())}

_ws = re.compile(r"\s+")
_nonalnum = re.compile(r"[^a-z0-9 ]+")
_punct_keep_num = re.compile(r"[^\w\s/-]", re.UNICODE)


def strip_accents_latin(s: str) -> str:
    """Remove accents from Latin text but leave other scripts untouched."""
    out = []
    for ch in s:
        if ord(ch) < 0x0250 or (0x1E00 <= ord(ch) < 0x2000):
            d = unicodedata.normalize("NFKD", ch)
            out.append("".join(c for c in d if not unicodedata.combining(c)))
        else:
            out.append(ch)
    return "".join(out)


def has_non_latin(s: str) -> bool:
    return any(ord(ch) > 0x024F and not unicodedata.combining(ch) for ch in s)


def romanize(s: str) -> str:
    """Full transliteration to ASCII (used for Indic-script names)."""
    return unidecode(s)


def basic_clean(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = s.replace("&", " and ").replace("+", " plus ")
    s = strip_accents_latin(s)
    s = unidecode(s)  # transliterate leftover non-Latin (Indic) so everything is ascii
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return _ws.sub(" ", s).strip()


# ---- names ----------------------------------------------------------------- #
_dom = re.compile(r"\.(com|in|net|org|co|io|biz|info|fr)\b")
_id_tag = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)", re.I)
_bracket_legal = re.compile(r"[\[\(\{]\s*(llc|inc|ltd|limited|pvt|corp|llp|plc|pc|lp)\s*[\]\)\}]", re.I)
_tok_split = re.compile(r"[^\s\.,;:\-/()\[\]{}'\"&+#!?|]+", re.UNICODE)

# Token-level transliteration dictionary learned from the training pairs
# (see translit.py).  Loaded lazily from translit_dict.json next to this file.
_TRANSLIT: dict | None = None


def _load_translit() -> dict:
    global _TRANSLIT
    if _TRANSLIT is None:
        import json
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translit_dict.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                _TRANSLIT = json.load(f)
        else:
            _TRANSLIT = {}
    return _TRANSLIT


# Rule-based fallback for common trade words (shop types) that do not occur in the
# training vocabulary.  Purely linguistic transliterations, grouped by concept.
_FALLBACK_GROUPS = {
    "motors": "मोटर्स ಮೋಟಾರ್ಸ್ మోటార్స్ மோட்டார்ஸ் মোটরস મોટર્સ മോട്ടോഴ്സ് ਮੋਟਰਜ਼ ମୋଟର୍ସ୍",
    "stores": "स्टोर्स ಸ್ಟೋರ್ಸ್ స్టోర్స్ ஸ்டோர்ஸ் স্টোর্স સ્ટોર્સ സ്റ്റോർസ് ਸਟੋਰਜ਼ ଷ୍ଟୋର୍ସ୍",
    "agencies": "एजेंसीज ಏಜೆನ್ಸೀಸ್ ఏజెన్సీస్ ஏஜென்சீஸ் এজেন্সিজ એજન્સીઝ ഏജൻസീസ് ਏਜੰਸੀਜ਼ ଏଜେନ୍ସିଜ୍",
    "jewellers": "ज्वेलर्स ಜ್ಯುವೆಲರ್ಸ್ జ్యువెలర్స్ நகைக்கடை জুয়েলার্স જ્વેલર્સ ജ്വല്ലേഴ്സ് ਜਵੈਲਰਜ਼ ଜୁଏଲର୍ସ୍",
    "general": "जनरल ಜನರಲ್ జనరల్ ஜெனரல் জেনারেল જનરલ ജനറൽ ਜਨਰਲ ଜେନେରାଲ୍",
    "bakery": "बेकरी ಬೇಕರಿ బేకరీ பேக்கரி বেকারি બેકરી ബേക്കറി ਬੇਕਰੀ ବେକେରୀ",
    "provision": "प्रोविजन ಪ್ರೊವಿಷನ್ ప్రొవిజన్ புரொவிஷன் প্রভিশন પ્રોવિઝન പ്രൊവിഷൻ ਪ੍ਰੋਵੀਜ਼ਨ ପ୍ରୋଭିଜନ୍",
    "sweets": "स्वीट्स ಸ್ವೀಟ್ಸ್ స్వీట్స్ இனிப்பகம் সুইটস સ્વીટ્સ സ്വീറ്റ്സ് ਸਵੀਟਸ ସ୍ୱିଟ୍ସ୍",
    "pharmacy": "फार्मेसी ಫಾರ್ಮಸಿ ఫార్మసీ மருந்தகம் ফার্মেসি ફાર્મસી ഫാർമസി ਫਾਰਮੇਸੀ ଫାର୍ମାସି",
    "medicals": "मेडिकल्स ಮೆಡಿಕಲ್ಸ್ మెడికల్స్ மெடிக்கல்ஸ் মেডিক্যালস મેડિકલ્સ മെഡിക്കൽസ് ਮੈਡੀਕਲਜ਼ ମେଡିକାଲ୍ସ୍",
    "automobiles": "ऑटोमोबाइल्स ಆಟೋಮೊಬೈಲ್ಸ್ ఆటోమొబైల్స్ ஆட்டோமொபைல்ஸ் অটোমোবাইলস ઓટોમોબાઇલ્સ ഓട്ടോമൊബൈൽസ് ਆਟੋਮੋਬਾਈਲਜ਼ ଅଟୋମୋବାଇଲ୍ସ୍",
    "traders": "ट्रेडर्स ಟ್ರೇಡರ್ಸ್ ట్రేడర్స్ வர்த்தகர்கள் ট্রেডার্স ટ્રેડર્સ ട്രേഡേഴ്സ് ਟ੍ਰੇਡਰਜ਼ ଟ୍ରେଡର୍ସ୍",
    "hardware": "हार्डवेयर ಹಾರ್ಡ್‌ವೇರ್ హార్డ్‌వేర్ ஹார்டுவேர் হার্ডওয়্যার હાર્ડવેર ഹാർഡ്‌വെയർ ਹਾਰਡਵੇਅਰ ହାର୍ଡୱେୟାର୍",
    "garments": "गारमेंट्स ಗಾರ್ಮೆಂಟ್ಸ್ గార్మెంట్స్ கார்மென்ட்ஸ் গার্মেন্টস ગારમેન્ટ્સ ഗാർമെന്റ്സ് ਗਾਰਮੈਂਟਸ ଗାର୍ମେଣ୍ଟସ୍",
    "steel": "स्टील ಸ್ಟೀಲ್ స్టీల్ ஸ்டீல் স্টিল સ્ટીલ സ്റ്റീൽ ਸਟੀਲ ଷ୍ଟିଲ୍",
    "electronics": "इलेक्ट्रॉनिक्स ಎಲೆಕ್ಟ್ರಾನಿಕ್ಸ್ ఎలక్ట్రానిక్స్ எலெக்ட்ரானிக்ஸ் ইলেকট্রনিক্স ઇલેક્ટ્રોનિક્સ ഇലക്ട്രോണിക്സ് ਇਲੈਕਟ੍ਰਾਨਿਕਸ ଇଲେକ୍ଟ୍ରୋନିକ୍ସ୍",
    "furniture": "फर्नीचर ಫರ್ನಿಚರ್ ఫర్నిచర్ ஃபர்னிச்சர் ফার্নিচার ફર્નિચર ഫർണിച്ചർ ਫਰਨੀਚਰ ଫର୍ନିଚର୍",
    "restaurant": "रेस्टोरेंट ರೆಸ್ಟೋರೆಂಟ್ రెస్టారెంట్ உணவகம் রেস্টুরেন্ট રેસ્ટોરન્ટ റെസ്റ്റോറന്റ് ਰੈਸਟੋਰੈਂਟ ରେଷ୍ଟୁରାଣ୍ଟ୍",
    "textiles": "टेक्सटाइल्स ಟೆಕ್ಸ್‌ಟೈಲ್ಸ್ టెక్స్‌టైల్స్ ஜவுளி টেক্সটাইলস ટેક્સટાઇલ્સ ടെക്സ്റ്റൈൽസ് ਟੈਕਸਟਾਈਲਜ਼ ଟେକ୍ସଟାଇଲ୍ସ୍",
}
FALLBACK_TRANSLIT = {tok: eng for eng, toks in _FALLBACK_GROUPS.items() for tok in toks.split()}


def transliterate_name(s: str) -> str:
    """Replace Indic-script tokens by their learned English equivalents.

    Tokens missing from the dictionary fall back to unidecode later."""
    d = _load_translit()
    out = []
    for t in _tok_split.findall(s):
        tl = t.lower()
        if has_non_latin(tl):
            out.append(d.get(tl) or FALLBACK_TRANSLIT.get(tl) or t)
        else:
            out.append(t)
    return " ".join(out)


def norm_name(raw: str) -> dict:
    """Return several normalised views of a business name."""
    if raw is None:
        raw = ""
    s = raw.strip()
    nonlat = has_non_latin(s)
    if nonlat:
        s = transliterate_name(s)
    s = _id_tag.sub(" ", s)
    s = _bracket_legal.sub(r" \1 ", s)
    s = _dom.sub(" ", s.lower())
    s = s.replace("#", " ")
    full = basic_clean(s)
    toks = full.split()
    # canonicalise legal tokens
    can = []
    legal = []
    for t in toks:
        if t in LEGAL_MAP:
            legal.append(LEGAL_MAP[t])
        else:
            can.append(t)
    # merged legal forms like "pvtltd"
    core_tokens = [t for t in can if t not in LEGAL_TOKENS]
    core = " ".join(core_tokens)
    sig = " ".join(t for t in core_tokens if t not in FILLER_TOKENS) or core
    return {
        "name_full": full,
        "name_core": core,
        "name_sig": sig,
        "name_legal": " ".join(sorted(set(legal))),
        "name_nospace": core.replace(" ", ""),
        "name_sorted": " ".join(sorted(core_tokens)),
        "name_nonlatin": nonlat,
    }


# ---- addresses --------------------------------------------------------------- #
_lead_zero = re.compile(r"\b0+(\d)")
_ordinal = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_city_suffix = re.compile(r"\b(city|town|cdp|borough|village|township|county)\b")
_null = re.compile(r"<null>|\bn/a\b|\bnull\b|\bnone\b|\bnan\b")


def norm_address(raw: str) -> dict:
    if raw is None:
        raw = ""
    s = raw.lower()
    for k, v in INDIC_STATE.items():
        if k in s:
            s = s.replace(k, " " + v + " ")
    s = _null.sub(" ", s)
    s = s.replace("#", " ")
    s = strip_accents_latin(s)
    s = unidecode(s)
    s = s.replace("&", " and ")
    # split numbers glued to letters: "8045a" -> "8045 a", "1st" handled by ordinal
    s = _ordinal.sub(r"\1", s)
    s = re.sub(r"[^a-z0-9/\-\s]", " ", s)
    s = re.sub(r"[/\-]+", " ", s)
    s = _lead_zero.sub(r"\1", s)
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s)
    s = re.sub(r"([a-z])(\d)", r"\1 \2", s)
    toks = s.split()
    out = []
    for t in toks:
        if t in ADDR_ABBR:
            r = ADDR_ABBR[t]
            if r:
                out.extend(r.split())
        else:
            out.append(t)
    # expand state abbreviations (only for 2-letter tokens at the end-ish; we apply everywhere
    # but keep both forms via the "state" field)
    full = " ".join(out)
    # numbers
    nums = [t for t in out if t.isdigit()]
    alphas = [t for t in out if not t.isdigit()]
    # collapse "city" suffix words
    alpha_str = _city_suffix.sub(" ", " ".join(alphas))
    alpha_str = _ws.sub(" ", alpha_str).strip()
    # state detection: any token that is a known abbreviation or a full state name
    state = ""
    joined = " " + alpha_str + " "
    for fullname in FULL_TO_ABBR:
        if " " + fullname + " " in joined:
            state = fullname
            joined = joined.replace(" " + fullname + " ", " ")
            break
    if not state:
        for t in reversed(alpha_str.split()):
            if t in STATE_ABBR:
                state = STATE_ABBR[t]
                joined = re.sub(r" %s " % re.escape(t), " ", joined, count=1)
                break
    alpha_wo_state = _ws.sub(" ", joined).strip()
    return {
        "addr_full": full,
        "addr_alpha": alpha_wo_state,
        "addr_nums": " ".join(nums),
        "addr_first_num": nums[0] if nums else "",
        "addr_state": state,
        "addr_empty": len(out) == 0,
    }


if __name__ == "__main__":
    tests = [
        ("Summitanimalhospital.Com", "14602 Orchard Ridge Rd, Hancock, Maryland"),
        ("Dynamic Mortgage Brands Ínc", "001375 QUILL AVE, BOONE, IA"),
        ("Tirumalagiri Enterprises Private-Ltd (ID: 59206)", "207, Block C, Shree, Balaji Enclave, X Rd, Ranga Reddy, Hyderabad, TG"),
        ("ಸದರ್ನ್ ಡೆವಲಪರ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "SRI BALAJI NILAYA, BANGALORE, ಕರ್ನಾಟಕ"),
        ("Wildlife Coalition [LLC]", "Crestview Dr, Toano, Virginia"),
        ("SCI Ptit Àmicale", "18 RUE JEN ZAY, Dunkerque, Nord"),
        ("#4443ocean", "12 Hen Towing Group | AZ, 8045A NATAL AVENUE, MESA CITY"),
    ]
    for n, a in tests:
        print(norm_name(n))
        print(norm_address(a))
        print()
