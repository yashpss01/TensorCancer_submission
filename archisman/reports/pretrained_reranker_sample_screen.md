# Pretrained multilingual reranker: bounded hard-pair screen

This is a **sampled pair-discrimination diagnostic on exposed development
cohorts**, not a new per-S1 macro-F0.5 result, untouched confirmation,
full-corpus test, France result, or Portal score. The frozen candidate lists
and packaged matcher were not changed.

We evaluated the pretrained [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
at revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Its publisher
describes it as a multilingual relevance reranker. It was downloaded and
scored locally on Apple MPS; **no Modal job or additional cloud spend** was
used. The original one-epoch MiniLM pilot in the structural matcher report
trained a new classification head, whereas this screen uses the publisher's
pretrained reranking head without fine-tuning.

For each direction, a rich matcher fitted on one exposed 10,000-S1 cohort
scored the other. Pairs with rich probability in `[0.05, 0.95]` were ordered
by a stable hash of their IDs; the first 1,000 were selected **without using
their labels**. Query and target name, address, and country were passed to
the reranker, with original scripts preserved. A separate process read only
the prepared input text, not the local truth file. Labels were opened for the
comparison after logits were saved. This selection emphasizes uncertain
pairs and does not represent the full candidate population.

| Exposed fit → sampled test | Sample pairs / positives | Rich AUC / average precision | Pretrained reranker AUC / average precision |
| --- | ---: | ---: | ---: |
| v1 → v2 | 1,000 / 475 | 0.8925 / 0.8836 | 0.6667 / 0.6007 |
| v2 → v1 | 1,000 / 445 | 0.8759 / 0.8512 | 0.6108 / 0.5047 |

On India pairs, reranker AUC was 0.7639 and 0.6985 in the two directions,
versus rich AUC 0.9001 and 0.8703. On US pairs it was 0.5714 and 0.5352,
versus 0.8864 and 0.8831. Scoring the 1,000 pairs took 27.6 and 26.2
seconds respectively after local model download. Full aggregate results:
[v1 → v2](pretrained_reranker_v1_to_v2_sample.json) and
[v2 → v1](pretrained_reranker_v2_to_v1_sample.json).

The pretrained relevance logits are markedly less discriminative than the
existing matcher on the same hard-pair samples. Do not scale this zero-shot
reranker to full candidates or interpret its AUC as a macro-F0.5 score.
Fine-tuning a different architecture remains a separate research question;
the confirmed 98.7707% reduced-pool macro-F0.5 remains the best result.
