# Reference dataset licence

`pdac_reference_dataset.json` is released under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

You may share and adapt it, including commercially, provided you give
appropriate credit and indicate whether changes were made.

## Contents

90 curated compounds with bioactivity annotations spanning pancreatic ductal
adenocarcinoma, non small cell lung cancer, antimicrobial resistance, MASLD and
negative controls. Each record carries its canonical SMILES, InChIKey, computed
descriptors, assay context, cell line and measurement type.

## Provenance and its limits

Values were curated from public sources, principally GDSC2, ChEMBL and the
published literature, and normalised into one schema. Assay conditions differ
between the original sources, so values are comparable within a disease
partition and not across partitions. The loader enforces that partitioning:
bacterial MIC values and hepatic receptor EC50 values are never mixed into a
PDAC cytotoxicity training target.

This is a small benchmark set assembled to exercise and validate the pipeline.
It is not a comprehensive bioactivity database, and a model trained on it
inherits its narrowness. The QSAR validation metrics printed by
`scripts/train_qsar.py` are the honest measure of what it supports.

## Citing

Cite the original data sources for any individual measurement. Cite this
repository for the curated and normalised collection.
