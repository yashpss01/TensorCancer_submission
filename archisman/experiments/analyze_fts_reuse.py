"""Count repeated FTS query expressions without executing any FTS searches."""

import argparse
import collections
import itertools
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from blocking import fields, rows  # noqa: E402
from evaluate import Retriever, quote  # noqa: E402
import evaluate  # noqa: E402
from inference_retrieval import Improved  # noqa: E402
from normalize import address, bigrams  # noqa: E402
from phonetic import grams  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", type=pathlib.Path, required=True)
    p.add_argument("--index-dir", type=pathlib.Path, required=True)
    p.add_argument("--rows", type=int, default=10000)
    p.add_argument("--report", type=pathlib.Path, required=True)
    a = p.parse_args()
    evaluate.ART = a.index_dir
    base = Retriever()
    rescue = Improved(a.index_dir)
    route_expr = collections.defaultdict(collections.Counter)
    expr = lambda col, ts: " OR ".join(col + ":" + quote(t) for t in ts)
    for q in itertools.islice(rows(a.source1), a.rows):
        ns, ads, gs = fields(q)
        n, ad, g = ns.split(), ads.split(), gs.split()
        rn = base.rare("name", n, 6)
        ra = base.rare("address", ad, 12)
        rg = base.rare("grams", g, 5)
        pg = grams(q["business_name"])
        rp = base.rare("phonetic", pg, 6)
        gp = ["(grams:" + quote(x) + " AND grams:" + quote(y) + ")"
              for i, x in enumerate(rg) for y in rg[i + 1:]]
        pp = ["(phonetic:" + quote(x) + " AND phonetic:" + quote(y) + ")"
              for i, x in enumerate(rp) for y in rp[i + 1:]]
        expressions = {
            "name": expr("name", rn), "address": expr("address", ra),
            "joint": "(" + expr("name", rn) + ") AND (" + expr("address", ra) + ")" if rn and ra else "",
            "char": " OR ".join(gp), "phonetic": " OR ".join(pp),
            "phonetic_joint": "(" + " OR ".join(pp) + ") AND (" + expr("address", ra) + ")" if pp and ra else "",
        }
        xa = rescue.rare("addr", address(q["business_address"]), 12)
        xb = rescue.rare("n2", bigrams(q["business_name"]), 6)
        xp = ["(n2:" + quote(x) + " AND n2:" + quote(y) + ")"
              for i, x in enumerate(xb) for y in xb[i + 1:]]
        expressions["address2"] = expr("addr", xa)
        expressions["script2"] = "(" + " OR ".join(xp) + ") AND (" + expr("addr", xa) + ")" if xp and xa else ""
        for route, sql in expressions.items():
            if sql:
                route_expr[route][sql] += 1
    result = {route: {"calls": sum(counter.values()), "unique": len(counter),
                      "repeat_calls": sum(value - 1 for value in counter.values()),
                      "max_frequency": max(counter.values())}
              for route, counter in route_expr.items()}
    a.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
