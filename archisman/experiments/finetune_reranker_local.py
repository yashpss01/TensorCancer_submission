"""Bounded local top-layer fine-tuning; score a separate unlabeled input.

The evaluation input contains no labels. This is an exposed-development
screen, not a frozen model promotion or fresh confirmation.
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

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


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
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--trainable-layers", type=int, default=4)
    p.add_argument("--checkpoint", type=pathlib.Path)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.max_length, args.trainable_layers) < 1 or args.output.exists():
        raise ValueError("Use positive sizes and a new output path")
    if args.checkpoint and args.checkpoint.exists():
        raise ValueError("Checkpoint output already exists")
    started = time.monotonic()
    torch.manual_seed(20260927)
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train = json.loads(args.training.read_text())
    evaluation = json.loads(args.evaluation_input.read_text())
    if train["cohort"] == evaluation["test_cohort"]:
        raise ValueError("Training and evaluation S1 cohorts must differ")
    token = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True,
                                          trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32,
    ).to(device)
    layers = model.roberta.encoder.layer
    if args.trainable_layers > len(layers):
        raise ValueError("More trainable layers requested than model contains")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in layers[-args.trainable_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_ids, train_masks = tokenize(token, train["records"], args.max_length)
    test_ids, test_masks = tokenize(token, evaluation["pairs"], args.max_length)
    labels = np.asarray([r["label"] for r in train["records"]], dtype=np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_ids), torch.from_numpy(train_masks),
                      torch.from_numpy(labels)),
        batch_size=args.batch_size, shuffle=True,
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=2e-5, weight_decay=.01)
    print("training", len(train["records"]), "pairs on", device,
          "trainable_parameters", trainable, "in", round(time.monotonic()-started, 1),
          "seconds", flush=True)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, (ids, masks, y) in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=ids.to(device),
                           attention_mask=masks.to(device)).logits.view(-1)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            if step % 100 == 0 or step == len(loader):
                print("epoch", epoch+1, "step", step, "/", len(loader),
                      "mean_loss", round(running/step, 4),
                      "seconds", round(time.monotonic()-started, 1), flush=True)
    logits = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(test_ids), args.batch_size*2):
            end = min(start+args.batch_size*2, len(test_ids))
            values = model(input_ids=torch.as_tensor(test_ids[start:end], device=device),
                           attention_mask=torch.as_tensor(test_masks[start:end], device=device))
            logits.extend(values.logits.view(-1).float().cpu().numpy().tolist())
    result = {
        "scope": "local top-layer fine-tuned reranker on exposed-development data",
        "model": MODEL_NAME,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "training_cohort": train["cohort"],
        "test_cohort": evaluation["test_cohort"],
        "training_pairs": len(train["records"]),
        "training_positives": int(labels.sum()),
        "evaluation_pairs": len(evaluation["pairs"]),
        "evaluation_input_sha256": hashlib.sha256(args.evaluation_input.read_bytes()).hexdigest(),
        "trainable_layers": args.trainable_layers,
        "trainable_parameters": trainable,
        "epochs": args.epochs,
        "device": str(device),
        "logits": logits,
        "seconds": time.monotonic()-started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        trainable_names = {name for name, parameter in model.named_parameters()
                           if parameter.requires_grad}
        torch.save({"trainable_state": {
                        name: parameter.detach().cpu()
                        for name, parameter in model.named_parameters()
                        if name in trainable_names},
                    "training_cohort": train["cohort"],
                    "model": MODEL_NAME,
                    "trainable_layers": args.trainable_layers,
                    "max_length": args.max_length}, args.checkpoint)
    args.output.write_text(json.dumps(result)+"\n")
    print("saved", args.output, "in", round(result["seconds"], 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
