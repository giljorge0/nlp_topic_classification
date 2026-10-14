# model_handoff_extras

Unzip NEXT TO your existing `model_handoff/` folder (the first folder merges into it).

1. `model_handoff/datasets/baseline_all_fields/train_augmented_word_dropout.csv` (recommended main augmentation, p=0.05) and `train_augmented_sentence_dropout.csv` (optional, p=0.10): originals + appended copies, TRAIN rows of the primary split only. Use for Experiment 4. `model_handoff/augmentation_check.json` records the automatic checks (all passed). The non-augmented files in your existing `model_handoff/` were verified byte-identical to the ones this was built from.
2. `model_handoff_group_aware/`: a complete second hand-off built from the optional group-aware split (own split manifests, datasets, class weights, vocabularies re-fitted on that split's train partition, plus its augmented files for `baseline_all_fields`). Use it only for the final robustness repeat. Its `handoff_manifest.json` lists SHA-256 checksums of every file.

Nothing here claims augmentation helps; that needs your model results.

## Revision 2 (sentence-dropout separator fix)
`train_augmented_sentence_dropout.csv` (primary and group-aware) was regenerated. The first version rebuilt the transcription with `" ".join(...)`, which silently rewrote the
separators between ALL kept sentences (e.g. `spells., She` -> `spells. She`; glued `.,2` -> `. 2`). Now the original separator that followed each retained sentence is kept,
so only the dropped sentences (and the separator after each) disappear. The same copies, sentences dropped, seeds, row counts (3950 primary / 3951 group-aware) and
per-class counts as before; only the text of the copies changed. `train_augmented_word_dropout.csv`, `augmentation_check.json`, all split manifests and all other files are byte-identical
to revision 1. In `model_handoff_group_aware/handoff_manifest.json` only the SHA-256/size entry of the regenerated file was updated. Discard revision 1 of this zip.
