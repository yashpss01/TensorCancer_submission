"""Score prepared hard pairs with a local pretrained multilingual reranker.

This process reads only input.json, never truth labels or training files.
"""

import argparse
import hashlib
import json
import pathlib
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    if args.batch_size < 1 or args.output.exists():
        raise ValueError("Use positive batch size and a new output path")
    started = time.monotonic()
    prepared = json.loads(args.input.read_text())
    pairs = prepared["pairs"]
    if len(pairs) != prepared["sample_pairs"]:
        raise ValueError("Prepared pair count differs")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True,
                                              trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, local_files_only=True, trust_remote_code=False,
        dtype=torch.float16 if device.type == "mps" else torch.float32,
    ).to(device).eval()
    print("model loaded on", device, "in", round(time.monotonic()-started, 1),
          "seconds", flush=True)
    logits = []
    with torch.inference_mode():
        for start in range(0, len(pairs), args.batch_size):
            end = min(start+args.batch_size, len(pairs))
            batch = tokenizer([p["query"] for p in pairs[start:end]],
                              [p["target"] for p in pairs[start:end]],
                              padding=True, truncation=True, max_length=192,
                              return_tensors="pt")
            result = model(**{key: value.to(device) for key, value in batch.items()})
            logits.extend(result.logits.view(-1).float().cpu().numpy().tolist())
            if end % 200 < args.batch_size or end == len(pairs):
                print("reranked", end, "in", round(time.monotonic()-started, 1),
                      "seconds", flush=True)
    report = {
        "scope": "pretrained reranker logits for a bounded, label-blind exposed-development sample",
        "model": MODEL_NAME,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "device": str(device), "sample_pairs": len(pairs),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "logits": logits,
        "seconds": time.monotonic()-started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report)+"\n")
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
