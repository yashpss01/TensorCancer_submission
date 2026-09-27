"""Count development candidate pairs indistinguishable by normalized text."""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    root = args.source_root
    sys.path.insert(0, str(root / "code/business_entity_resolution/round2"))
    from improve import select
    from normalize import address
    from common import tokens

    fresh = root / "artifacts/blocking_round4/fresh"
    art = root / "artifacts/matching_round5"
    truth = json.loads((fresh / "truth.json").read_text())
    offsets = np.load(art / "validation/offsets.npy")
    y = np.load(art / "validation/y.npy", mmap_mode="r").astype(bool)
    prob = 0.4 * np.load(art / "validation_group_extra_prob.npy") + 0.6 * np.load(art / "validation_augmented_0p15_prob.npy")
    old = root / "artifacts/blocking_round3/fresh/index.sqlite"
    db = sqlite3.connect(f"file:{old}?mode=ro", uri=True)
    records = {mid: (name, addr, country) for mid, name, addr, country in
               db.execute("SELECT id,name,address,country FROM records")}
    db.close()
    result = {
        "scope": "exposed 10k development, 409141-target reduced pool; text-only normalized collisions",
        "groups": 0,
        "raw_mixed_groups": 0,
        "normalized_mixed_groups": 0,
        "raw_mixed_positive_pairs": 0,
        "normalized_mixed_positive_pairs": 0,
        "raw_mixed_negative_pairs": 0,
        "normalized_mixed_negative_pairs": 0,
        "baseline_fn_in_normalized_collision": 0,
        "baseline_fp_in_normalized_collision": 0,
        "total_baseline_candidate_fn": int((y & (prob < 0.74)).sum()),
        "total_baseline_candidate_fp": int((~y & (prob >= 0.74)).sum()),
    }
    with (fresh / "improved.jsonl").open() as handle:
        for index, line in enumerate(handle):
            if index >= len(offsets) - 1:
                break
            row = json.loads(line)
            mids = select(row, 0.5, 16)
            lo, hi = offsets[index:index+2]
            assert hi - lo == len(mids) and row["id"] in truth
            raw_groups, norm_groups = {}, {}
            for j, mid in enumerate(mids):
                name, addr, country = records[mid]
                raw_key = (name, addr, country)
                norm_key = (" ".join(tokens(name)), " ".join(address(addr)), country)
                raw_groups.setdefault(raw_key, []).append(j)
                norm_groups.setdefault(norm_key, []).append(j)
            result["groups"] += 1
            for kind, groups in (("raw", raw_groups), ("normalized", norm_groups)):
                mixed = [members for members in groups.values()
                         if any(y[lo+j] for j in members) and any(not y[lo+j] for j in members)]
                if not mixed:
                    continue
                result[f"{kind}_mixed_groups"] += 1
                result[f"{kind}_mixed_positive_pairs"] += sum(y[lo+j] for members in mixed for j in members)
                result[f"{kind}_mixed_negative_pairs"] += sum(not y[lo+j] for members in mixed for j in members)
                if kind == "normalized":
                    result["baseline_fn_in_normalized_collision"] += sum(
                        bool(y[lo+j] and prob[lo+j] < 0.74) for members in mixed for j in members)
                    result["baseline_fp_in_normalized_collision"] += sum(
                        bool(not y[lo+j] and prob[lo+j] >= 0.74) for members in mixed for j in members)
    result = {key: int(value) if isinstance(value, np.integer) else value
              for key, value in result.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
