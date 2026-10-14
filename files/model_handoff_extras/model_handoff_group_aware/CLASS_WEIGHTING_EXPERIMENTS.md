# Class-weighting experiments (recommendations for the existing model code)

I did not run any of these. Weights below are **candidates**, not tuned values; do not call one optimal before it has been trained and evaluated.

## Definitions (computed from the PRIMARY split's TRAIN partition class counts)
`freq_c = count_c / N_train` (a proportion). Every vector is normalised so the **mean over the 12 classes is 1**.
1. `none`: all 1.
2. `inverse_frequency`: `1/freq_c`, normalised.
3. `inverse_sqrt_frequency`: `1/sqrt(freq_c)`, normalised.
4. `interpolated_alpha_<a>`: `weight_c(alpha) = normalised((1 - alpha) + alpha / freq_c)`; alpha=0 -> equal weights, alpha=1 -> inverse frequency.

**Caution on the interpolation:** because `freq_c` is a proportion (1/freq ranges ~3-106), the `1 - alpha` term is small next to `alpha/freq_c`, so the effective strength rises quickly with alpha rather than linearly:

| strategy                |   min_weight |   max_weight |   max_over_min |   alpha |
|:------------------------|-------------:|-------------:|---------------:|--------:|
| interpolated_alpha_0    |        1     |        1     |          1     |    0    |
| none                    |        1     |        1     |          1     |  nan    |
| inverse_sqrt_frequency  |        0.365 |        2.177 |          5.967 |  nan    |
| interpolated_alpha_0.25 |        0.194 |        3.541 |         18.272 |    0.25 |
| interpolated_alpha_0.5  |        0.138 |        3.717 |         26.929 |    0.5  |
| interpolated_alpha_0.75 |        0.118 |        3.781 |         32.13  |    0.75 |
| inverse_frequency       |        0.107 |        3.814 |         35.6   |  nan    |
| interpolated_alpha_1    |        0.107 |        3.814 |         35.6   |    1    |

(`max_over_min` = strongest / weakest class weight.) If you want a *linear* blend of the two normalised vectors, say so and I will change the formula.

Per-class values: `weights/class_weights_primary_split_wide.csv` (also `.json`; a version for the group-aware split's train partition is in `class_weights_long.csv`, column `basis`).

## Interface the training code needs
A float vector `w` of length 12 aligned to the model's class indices (map by class NAME; `label_to_index.json` is only this repo's order). Typical use: PyTorch `CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))`; Keras `class_weight={idx: w_i}`. If the code does not expose this, the minimal change is one optional argument (e.g. `--class-weights path.json`) passed to the loss. I did not touch their code.

## Suggested first-pass table (keep EVERYTHING else fixed)
Fixed: architecture, preprocessing variant (start with `baseline_all_fields`), `splits/split_manifest_primary_stratified.csv`, seed, optimizer, training budget, no augmentation.

| run | class weights | notes |
|---|---|---|
| 1 | none | reference |
| 2 | inverse_sqrt_frequency | |
| 3 | inverse_frequency | |
| 4-8 (only if 2/3 look promising) | interpolated alpha 0, 0.25, 0.5, 0.75, 1.0 | alpha=0 duplicates run 1, alpha=1 duplicates run 3 (skip) |

Primary objective: **validation macro-F1**; also record accuracy and **every class's F1** (Dermatology and Psychiatry-Psychology have the fewest validation examples, so their F1 is noisy). Early stopping on a predeclared metric, preferably macro-F1 if the code supports it reliably. Run more seeds only for the 1-2 best configurations, if compute allows. Do not change weights, architecture, input fields and augmentation at the same time; vary one factor per table.

Also report each run on the subsets in `splits/validation_subsets_primary.csv` (complete-input-conflict / consistent-duplicate / no complete-input twin in train). Check that the course's automatic criteria (accuracy > 49% and per-class F1 >= 25%) hold on validation.

## What weights cannot do
Class weights only change the optimisation trade-off. They cannot resolve notes whose complete inputs are identical but whose labels differ (171 records in 82 groups): at most one label per identical input can be right.

## Duplicate-group weights (diagnostic, separate from class weights)
`weights/duplicate_group_sample_weights_primary_train.csv`: per train record `1/(train records in its transcription duplicate group)`, mean 1. If the code supports sample weights, combine as `sample_weight_i = class_weight[label_i] * group_weight_i`, then optionally rescale to mean 1. Evaluate it as its own experiment; do not mix with the class-weight comparison.

No oversampling or duplication of minority records is done anywhere in the default data.
