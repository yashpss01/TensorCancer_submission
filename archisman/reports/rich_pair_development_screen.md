# Rich pair reranker: exposed-cohort development screen

The original blocker and its candidate lists were held fixed. A label-blind export reproduced every frozen pair probability and added the original name/address/group features, global core-name frequencies, and five RapidFuzz/Unidecode similarities. This is a matcher-only experiment on the two **already exposed** 10,000-S1 reduced-pool batches; it is not a fresh confirmation or a full-corpus test.

Each direction fitted on one exposed batch, calibrated its threshold by entity-disjoint crossfit within that training batch, and evaluated on the other exposed batch.

| Model features | Train v1 → test v2 F0.5 | India v2 | Train v2 → test v1 F0.5 | India v1 |
| --- | ---: | ---: | ---: | ---: |
| Six core-frequency features | 98.7290% | 98.4346% | 98.7821% | 98.5992% |
| Original rich features + core frequency | 98.7273% | 98.4645% | **98.8224%** | **98.6398%** |
| Rich features + Unicode/RapidFuzz | **98.7564%** | **98.4724%** | 98.7959% | 98.6375% |

The richer models add only a few hundredths of a percentage point over the recalibrated six-feature control, and the ordering differs by direction. The score remains far below 99.5%. This is evidence that wider text features alone are unlikely to close the gap, though combining the two exposed cohorts may give a small incremental gain.

For a single frozen confirmation, we fitted the preselected rich-plus-RapidFuzz configuration on both exposed batches (20,000 S1, 1,003,680 candidate pairs). Entity-disjoint crossfit across those batches selected threshold 0.775 and measured 98.8034% macro F0.5 overall, 98.5944% India, and 98.9441% US. This crossfit estimate is development evidence because both batches have already been inspected. The model and threshold are frozen in `experiments/checkpoints/rich_pair_v1v2/manifest.json` before opening the reserved next cohort's truth. A later result must be reported separately, even if it is worse.

The model uses only text, country, global unlabeled name frequencies, and frozen-model scores. It does not use entity IDs as predictive features, truth labels at inference, or Portal test output. Exporting these richer features took about 242 seconds per 10,000 S1 on the reduced pool, in addition to the expensive candidate retrieval. This research path has **not** improved production inference speed.

Saved development results: `work/rich_pair_model_screen.json`; frozen checkpoint: `experiments/checkpoints/rich_pair_v1v2/`. The raw feature matrices are ignored local work artifacts and are not needed to reproduce the reported scores from the documented scripts and source data.
