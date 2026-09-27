# One target, one Source 1 owner: structural development screen

The brief describes Source 1 as deduplicated. Streaming the **entire supplied
training ground truth** verified the corresponding invariant: 2,206,821
Source 1 rows contain 7,638,365 true links to 7,638,365 distinct Source 2/3
target IDs; **no target ID has two true Source 1 owners**. The input file's
SHA-256 is saved with the numeric result. This check read an aggregate property
from all training labels, including rows previously reserved for later work.
Their individual outcomes were not inspected, but they are **not strictly
label-sealed for testing this ownership hypothesis**. Do not call a later
training-row evaluation of this rule untouched confirmation.

We tested a label-blind postprocessor on the same fixed candidate lists and
rich pair scores. When multiple S1 rows accepted the same target above the
rich model's existing threshold, it kept only the highest-probability owner,
breaking exact ties by S1 ID. The rule did not inspect truth or add/remove
candidates. Training data labels were opened only after its choices were
fixed, to measure the effect. The three 10,000-S1 cohorts below were already
exposed; even the former final cohort is **not** an untouched confirmation of
this new rule.

| Exposed fit → test | Duplicate predicted targets | Removed true / false links | Overall macro F0.5 before → after | India before → after | US before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 → v2 | 9 | 3 / 6 | 98.7564% → 98.7674% | 98.4724% → 98.4946% | 98.9457% → 98.9493% |
| v2 → v1 | 8 | 0 / 8 | 98.7959% → 98.8244% | 98.6375% → 98.6858% | 98.9035% → 98.9186% |
| frozen v1+v2 → former final | 14 | 0 / 14 | 98.7707% → 98.7947% | 98.5541% → 98.5780% | 98.9204% → 98.9446% |

Paired S1-bootstrap 95% intervals for the overall gain, in percentage
points, were +0.0003 to +0.0267, +0.0074 to +0.0563, and +0.0085 to
+0.0430 respectively. The rule is logically consistent with all provided
training labels, but a model can rank a false claimant above a true one:
three true links were removed in the first direction. Its measured gains
are only **+0.0110 to +0.0285 points** on these reduced, positive-enriched
target pools, far short of the 99.5% goal. A full-corpus test can have many
more competing S1 rows and could behave differently.

**Decision:** preserve the frozen packaged matcher. This is a small,
reproducible structural development gain, not a fresh score or a reason to
claim the target. If combined with a materially stronger decision method,
freeze the complete rule and threshold before evaluation; a truly independent
labeled set or Portal result would be needed to call this ownership rule
untouched confirmation. Full counts, country scores, paired intervals, and the training
truth digest are in
[target_owner_uniqueness_screen.json](target_owner_uniqueness_screen.json).
