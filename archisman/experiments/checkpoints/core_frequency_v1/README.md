# Frozen development checkpoint

This six-feature name-frequency calibrator was trained on the already exposed
fresh-v1 10,000-business reduced-pool cohort. Its 0.70 threshold was chosen
from a two-fold, entity-disjoint development screen, then frozen before the
next disjoint cohort's matching labels were scored. The manifest records input
digests and the model hash.

It is **experimental**, not the submitted inference model. The two-fold
development macro F0.5 was 98.7723%; that figure is selection-optimistic and
does not establish the 99.5% goal, full-corpus performance, or France
performance. The next independent validation result must be reported before
any deployment decision.
