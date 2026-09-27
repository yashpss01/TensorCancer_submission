"""Bounded, local-MPS text-pair pilot on exposed development cohorts.

Train on entity-disjoint v1 development groups; calibrate on v1 holdout
groups; evaluate on exposed v2. This is not a fresh confirmation or a
submission inference implementation. No Portal data or cloud is used.
"""

import argparse
import csv
import hashlib
import json
import pathlib
import sqlite3
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from unidecode import unidecode


def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def load_data(root, training):
    source = read_tsv(root / "source1.tsv")
    truth = read_tsv(root / "truth.tsv")
    if [r["entity_id"] for r in source] != [r["source1_entity_id"] for r in truth]:
        raise ValueError("Source/truth order mismatch")
    frame = pl.read_parquet(root / "frozen_pair_scores.parquet")
    group_index = {row["entity_id"]: i for i, row in enumerate(source)}
    groups = np.array([group_index[sid] for sid in frame["s1_id"].to_list()], dtype=np.int32)
    mids = frame["target_id"].to_list()
    p0 = frame["frozen_probability"].to_numpy().astype(np.float32)
    actual = [set(filter(None, row["matched_entity_ids"].split(","))) for row in truth]
    labels = np.array([mid in actual[i] for mid, i in zip(mids, groups)], dtype=bool)
    fold = np.array([hashlib.sha256(row["entity_id"].encode()).digest()[0] % 5
                     for row in source], dtype=np.int8)
    selected = (p0 >= .01) | (labels if training else False)
    if training:
        easy_draw = np.random.default_rng(20260927).random(len(p0)) < .01
        selected |= easy_draw
    db = sqlite3.connect(f"file:{root/'bounded_index/index.sqlite'}?mode=ro", uri=True)
    wanted = sorted({mids[j] for j in np.where(selected)[0]})
    targets = {}
    for start in range(0, len(wanted), 900):
        batch = wanted[start:start+900]
        query = "SELECT id,name,address FROM records WHERE id IN ("+",".join("?"*len(batch))+")"
        targets.update({mid: (name, addr) for mid, name, addr in db.execute(query, batch)})
    db.close()
    if len(targets) != len(wanted):
        raise ValueError("Missing selected target")
    indices = np.where(selected)[0]
    a, b = [], []
    for j in indices:
        q = source[groups[j]]
        tname, taddr = targets[mids[j]]
        a.append(q["country"]+" name: "+unidecode(q["business_name"])+" address: "+unidecode(q["business_address"]))
        b.append("name: "+unidecode(tname)+" address: "+unidecode(taddr))
    return {"source": source, "truth": truth, "group": groups, "fold": fold,
            "labels": labels, "p0": p0, "indices": indices,
            "a": a, "b": b}


def tokenize(tokenizer, data, max_length):
    ids, masks = [], []
    for start in range(0, len(data["a"]), 512):
        encoded = tokenizer(data["a"][start:start+512], data["b"][start:start+512],
                            truncation=True, max_length=max_length,
                            padding="max_length", return_tensors="np")
        ids.append(encoded["input_ids"])
        masks.append(encoded["attention_mask"])
    return np.concatenate(ids), np.concatenate(masks)


def predict(model, device, ids, masks, batch_size):
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            a = torch.as_tensor(ids[start:start+batch_size], device=device)
            b = torch.as_tensor(masks[start:start+batch_size], device=device)
            logits = model(input_ids=a, attention_mask=b).logits
            result.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(result)


def entity_scores(chosen, data):
    n = len(data["source"])
    tp = np.bincount(data["group"][chosen & data["labels"]], minlength=n)
    fp = np.bincount(data["group"][chosen & ~data["labels"]], minlength=n)
    truth_count = np.array([len(set(filter(None, row["matched_entity_ids"].split(","))))
                            for row in data["truth"]], dtype=np.int32)
    return np.where(truth_count == 0, (fp == 0).astype(float),
                    1.25*tp / np.maximum(tp+.25*truth_count+fp, 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=96)
    args = p.parse_args()
    started = time.monotonic()
    torch.manual_seed(20260927)
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, local_files_only=True).to(device)
    v1 = load_data(args.root / "work/fresh_10k_v1", training=True)
    v2 = load_data(args.root / "work/fresh_10k_v2", training=False)
    id1, mask1 = tokenize(tokenizer, v1, args.max_length)
    id2, mask2 = tokenize(tokenizer, v2, args.max_length)
    selected_group = v1["group"][v1["indices"]]
    train = v1["fold"][selected_group] != 0
    valid = ~train
    y1 = v1["labels"][v1["indices"]].astype(np.int64)
    tensor_data = TensorDataset(torch.from_numpy(id1[train]),
                                torch.from_numpy(mask1[train]),
                                torch.from_numpy(y1[train]))
    loader = DataLoader(tensor_data, batch_size=args.batch_size, shuffle=True)
    positives = int(y1[train].sum())
    negatives = int(train.sum()-positives)
    weights = torch.tensor([positives/max(negatives, 1), 1.], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.
        for step, (a, b, y) in enumerate(loader, 1):
            a, b, y = a.to(device), b.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=a, attention_mask=b).logits
            loss = F.cross_entropy(logits, y, weight=weights)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().cpu())
            if step % 100 == 0:
                print(f"epoch {epoch+1} step {step}/{len(loader)} loss {loss_total/step:.4f} elapsed {time.monotonic()-started:.1f}s", flush=True)
        print(f"epoch {epoch+1} done in {time.monotonic()-started:.1f}s", flush=True)
    p1 = np.zeros(len(v1["labels"]), dtype=np.float32)
    p1[v1["indices"][valid]] = predict(model, device, id1[valid], mask1[valid], args.batch_size*2)
    p2 = np.zeros(len(v2["labels"]), dtype=np.float32)
    p2[v2["indices"]] = predict(model, device, id2, mask2, args.batch_size*2)
    valid_groups = v1["fold"] == 0
    grid = []
    for blend in (0., .25, .5, .75, 1.):
        val_scores = (1-blend)*v1["p0"]+blend*p1
        for threshold in np.arange(.45, .951, .025):
            scores = entity_scores(val_scores>=threshold, v1)
            grid.append((float(scores[valid_groups].mean()), blend, float(round(threshold, 3))))
    best, blend, threshold = max(grid, key=lambda x:x[0])
    test_scores = (1-blend)*v2["p0"]+blend*p2
    test_entities = entity_scores(test_scores>=threshold, v2)
    country = np.array([row["country"] for row in v2["source"]])
    baseline = entity_scores(v2["p0"]>=.74, v2)
    hard = v2["p0"]>=.01
    report = {
        "scope": "v1 exposed entity-disjoint train/calibration; v2 exposed development test; no fresh confirmation",
        "device": str(device), "epochs": args.epochs, "batch_size": args.batch_size,
        "max_length": args.max_length,
        "train_pairs": int(train.sum()), "validation_pairs": int(valid.sum()),
        "train_positive_pairs": positives, "train_negative_pairs": negatives,
        "v2_scored_pairs": len(v2["indices"]),
        "v2_hard_auc": float(roc_auc_score(v2["labels"][hard], p2[hard])),
        "selected_on_v1_validation": {"blend": blend, "threshold": threshold,
                                       "macro_f0_5": best},
        "v2_original_baseline": float(baseline.mean()),
        "v2_changed_overall": float(test_entities.mean()),
        "v2_changed_India": float(test_entities[country == "India"].mean()),
        "v2_changed_US": float(test_entities[country == "US"].mean()),
        "seconds": time.monotonic()-started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
