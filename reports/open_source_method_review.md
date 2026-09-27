# Open-source entity-resolution methods: fit to this challenge

This is an architecture review, **not** a score claim or a decision to
replace the frozen pipeline. The challenge brief permits only the provided
training data for matching. It prohibits external business-identity lookup,
registries, geocoding APIs, and internet data augmentation. Any final model
must be MIT/Apache-2.0 licensed and have at most 8 billion parameters. We
may read public architecture and code documentation, but must never send
challenge records to an external resolution service. The frozen rich matcher
still has **98.7707% per-S1 macro F0.5 on a reduced, positive-enriched 10k
confirmation pool**, below the 99.5% goal; no full-corpus/France/Portal score
has been measured.

## Comparison with published approaches

| Approach and primary source | Mechanism worth considering | Already tested here | Genuinely new, bounded test | Decision |
| --- | --- | --- | --- | --- |
| [Splink](https://moj-analytical-services.github.io/splink/): Fellegi–Sunter evidence, [term-frequency adjustments](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html), [business linkage example](https://moj-analytical-services.github.io/splink/demos/examples/duckdb_no_test/business_rates_match.html) | Calibrated match/non-match likelihood ratios by comparison level; rare agreements get more evidential weight | Name token IDF, global core-name frequency, fuzzy name/address features, and an XGBoost matcher already cover much of this signal | Tested a small cross-cohort likelihood-ratio comparator for joint name/address/missingness levels on the **same frozen candidates** | **Rejected comparator:** only 92.52% and 92.49% overall macro F0.5 in the two directions. This is our bounded implementation, not a benchmark of Splink itself. A whole-framework migration has no evidence yet of raising the score. |
| [Dedupe](https://docs.dedupe.io/en/latest/how-it-works/Matching-records.html), [variable definitions](https://docs.dedupe.io/en/latest/Variable-definition.html) | Active learning prioritizes blocker/classifier disagreement; explicit missing-value indicators and field interactions | Hard-negative mining, separate missing-address specialist, and group-context features have been screened without repeatable material gain | Error-stratified annotation/quality audit **only within provided labeled training rows** could test whether a systematic representation is missing | Do not solicit or infer new test labels; re-running a generic Dedupe pipeline is not a novel experiment. |
| [Ditto](https://github.com/megagonlabs/ditto) | Entity-matching sequence-pair training with explicit attribute serialization, domain-specific tags, and within-record augmentation | One-epoch MiniLM and a BGE relevance reranker, including top-four-layer local fine-tuning, were screened; the latter's blended F0.5 gain failed to repeat | A bounded, entity-matching-specific full-encoder training screen with name/address field markers and augmentation derived **only** from supplied training records | Most distinct untested model route. Evaluate on exposed, entity-disjoint cohorts first; do not promote from pair AUC alone. Any adopted weights need their own verified MIT/Apache-2.0 license and parameter count. |

The [Splink project](https://moj-analytical-services.github.io/splink/)
emphasizes that its likelihood model works best with multiple, reasonably
independent fields, and cautions against a lone bag-of-words name field. Our
name/address/country data are weaker than the richer business examples in
that documentation, so an off-the-shelf accuracy or speed claim would not
transfer to this dataset. Dedupe's missing-data indicators are already close
to our tested address-missing flags. Ditto's attribute serialization and
training augmentations are a materially different training recipe, although
our generic neural screens warn that an encoder does not automatically beat
the tabular model. These judgments are inferences from the cited designs and
our measured local screens, not published performance claims about our data.

## Fair-play and evaluation gate

The experimental [BGE reranker weights](https://huggingface.co/BAAI/bge-reranker-v2-m3)
list Apache-2.0 on their publisher's model card and contain 567,755,777
parameters by local safetensors-shape count. All scoring and fine-tuning was
local; no challenge records were uploaded. That model is **not adopted**:
its small forward development gain reversed on the other 10k cohort (see
[reranker screen](finetuned_reranker_development_screen.md)). A framework's
open-source license does **not** establish the license of pretrained model
weights; verify both separately before any future promotion and document
their provenance in the submission methodology.

The bounded likelihood-ratio comparison is now complete. It used four fixed
categorical evidence groups: name agreement plus core-name rarity, address
agreement/missingness, address-number evidence, and postal-code evidence.
Training-entity crossfit selected each threshold before evaluating the other
exposed cohort. It scored **92.5242% overall / 90.6154% India** from v1 → v2,
and **92.4949% / 90.6674%** in reverse, versus rich scores of 98.7564% and
98.7959% overall on those same candidates. Full counts, selected thresholds,
and learned weights are in
[likelihood_ratio_linkage_screen.json](likelihood_ratio_linkage_screen.json).
The simple conditionally independent evidence model discarded too much
discriminative detail here; this does not establish how a fully configured
Splink implementation would perform.

The remaining distinct research option is a bounded Ditto-style
entity-matching training recipe, rather than another generic relevance
reranker. A newly reserved disjoint 10k cohort must remain unopened
until a candidate rule and threshold are frozen. Neither research source
gives a defensible promise of reaching 99.5%.
