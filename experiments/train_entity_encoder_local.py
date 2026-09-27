"""Bounded Ditto-style field-pair encoder screen on exposed cohorts only.

Download of public model weights is allowed; challenge business records remain
local. The evaluation input is unlabeled. This script does not alter the
packaged matcher, create submission TSVs, or open a sealed cohort.
"""

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "distilbert/distilbert-base-multilingual-cased"
REVISION = "45c032ab32cc946ad88a166f7cb282f58c753c2e"


def tokenize(tokenizer, records, max_length):
    ids, masks = [], []
    for start in range(0, len(records), 512):
        batch = records[start:start+512]
        encoded = tokenizer([r["query"] for r in batch],
                            [r["target"] for r in batch],
                            padding="max_length", truncation=True,
                            max_length=max_length, return_tensors="np")
        ids.append(encoded["input_ids"])
        masks.append(encoded["attention_mask"])
    return np.concatenate(ids), np.concatenate(masks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training", type=pathlib.Path, required=True)
    p.add_argument("--evaluation-input", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-seconds", type=float, default=1200)
    args = p.parse_args()
    if (args.output.exists() or
            min(args.epochs, args.batch_size, args.max_length, args.max_seconds) <= 0):
        raise ValueError("Use positive limits and a new output path")
    started = time.monotonic()
    torch.manual_seed(20260927)
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train = json.loads(args.training.read_text())
    evaluation = json.loads(args.evaluation_input.read_text())
    if (train["cohort"] == evaluation["test_cohort"] or
            train["style"] != "fields" or evaluation["style"] != "fields"):
        raise ValueError("Expected distinct cohorts and field-marked inputs")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                               trust_remote_code=False)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[COL]", "[VAL]"]})
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, num_labels=1,
        ignore_mismatched_sizes=True, trust_remote_code=False,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_ids, train_masks = tokenize(tokenizer, train["records"], args.max_length)
    test_ids, test_masks = tokenize(tokenizer, evaluation["pairs"], args.max_length)
    labels = np.asarray([r["label"] for r in train["records"]], dtype=np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_ids), torch.from_numpy(train_masks),
                      torch.from_numpy(labels)),
        batch_size=args.batch_size, shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
    print("training", len(train["records"]), "pairs on", device,
          "trainable_parameters", trainable,
          "elapsed", round(time.monotonic()-started, 1), flush=True)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, (ids, masks, y) in enumerate(loader, 1):
            if time.monotonic()-started > args.max_seconds:
                raise TimeoutError("Bounded training time exceeded")
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids=ids.to(device),
                           attention_mask=masks.to(device)).logits.view(-1)
            loss = F.binary_cross_entropy_with_logits(output.float(), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach().cpu())
            if step % 100 == 0 or step == len(loader):
                print("epoch", epoch+1, "step", step, "/", len(loader),
                      "mean_loss", round(running/step, 4),
                      "seconds", round(time.monotonic()-started, 1), flush=True)
    model.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, len(test_ids), args.batch_size*2):
            if time.monotonic()-started > args.max_seconds:
                raise TimeoutError("Bounded scoring time exceeded")
            end = min(start+args.batch_size*2, len(test_ids))
            values = model(input_ids=torch.as_tensor(test_ids[start:end], device=device),
                           attention_mask=torch.as_tensor(test_masks[start:end], device=device))
            logits.extend(values.logits.view(-1).float().cpu().numpy().tolist())
    result = {
        "scope": "local exposed-cohort field-marked entity-matching encoder screen",
        "model": MODEL, "model_revision": REVISION,
        "training_cohort": train["cohort"],
        "test_cohort": evaluation["test_cohort"],
        "training_pairs": len(train["records"]),
        "training_positives": int(labels.sum()),
        "evaluation_pairs": len(evaluation["pairs"]),
        "evaluation_input_sha256": hashlib.sha256(args.evaluation_input.read_bytes()).hexdigest(),
        "trainable_parameters": trainable,
        "epochs": args.epochs,
        "device": str(device), "logits": logits,
        "seconds": time.monotonic()-started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result)+"\n")
    print("saved", args.output, "in", round(result["seconds"], 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
