# apertus_data

A Python package for managing the lifecycle of training datasets in a
traceable, schema-driven way — catalogue, build, hash, and track provenance.

## Overview

Each dataset is described by a YAML file in `catalogue/<dataset_id>.yaml`
that conforms to a single JSON Schema (`apertus_data/dataset_schema.yaml`,
the source of truth). The `Dataset` class loads this metadata, runs a
pinned per-dataset build script, hashes the produced files, and writes the
build entry back into the catalogue.

New datasets are proposed via a GitHub issue. The issue form is generated
from the schema, so adding or renaming a field touches one place.

## Layout

```
apertus_data/
├── apertus_data/
│   ├── dataset.py          # Dataset class — build, hash, persist
│   ├── build.py            # Builder discovery, signature contract, runner
│   ├── yaml.py             # YAML enrichment + JSON-Schema validation
│   ├── issue_template.py   # GitHub Issue Form generator
│   ├── dataset_schema.yaml # Single source of truth
│   ├── constants.py        # Paths, tokens, GitHub coordinates
│   └── utils.py            # Hashing, logging, helpers
├── builders/               # Per-dataset build scripts (`main(output_dir, logs_dir)`)
├── catalogue/              # Per-dataset YAML metadata
└── .github/ISSUE_TEMPLATE/ # Auto-generated issue forms
```

## Key concepts

**Catalogue.** Every dataset has a YAML at `catalogue/<id>.yaml` declaring
provenance (`owner`, `name`, `version`), classification (`tags`,
`modalities`, `licenses`, `formats`), and `build_requirements` — a pin on
the build script's GitHub URL and commit.

**Builders.** Each dataset has a build script at `builders/<id>.py`
exposing a `main(output_dir, logs_dir)` function. Scripts may call shared
helpers like `builders.default_hf_download.default_hf_download` for common
patterns.

**Runner contract.** Builders are never invoked directly. `Dataset.build()`
runs them through `build.run_builder`, which:

1. HEAD-checks the script on GitHub raw at the recorded commit;
2. `git checkout`s the local file to that commit if its history has drifted;
3. validates `main`'s signature (`(output_dir, logs_dir, *, **kwargs) -> None`);
4. attaches a per-run `FileHandler` writing to the dataset's `logs_dir`;
5. asserts `output_dir` is non-empty when the script returns.

**Provenance.** After each build, files in `data_dir` are hashed (Adler32
per file) and a single dataset digest is computed over the sorted per-file
hashes. The digest plus timestamp is appended to `build_history` and
persisted to the catalogue YAML.

**Schema-driven everything.** `dataset_schema.yaml` defines what's
required, what's nullable, and what enum values are allowed. It also
carries `x-git-issue-*` hints that drive the GitHub Issue Form layout, so
adding a field updates both validation and the contributor UX in one edit.

## Typical workflow

1. **Propose.** Open the "New dataset" GitHub issue and fill out the form.
2. **Validate.** Run the YAML pipeline to auto-generate `id`, normalize
   `version`, and check the schema:
   ```bash
   python -m apertus_data.yaml
   ```
3. **Author the builder.** Write `builders/<id>.py` exposing
   `main(output_dir, logs_dir)`, commit and push it, then record its commit
   in the YAML's `build_requirements.build_script_commit`.
4. **Build.**
   ```python
   from apertus_data.dataset import Dataset
   Dataset.from_id('owner___name___version').build()
   ```
   The runner verifies the script remotely, aligns the local file to the
   recorded commit, validates the signature, runs `main`, hashes the
   output, and records the build in the catalogue.

## Regenerating the issue form

```bash
python -m apertus_data.issue_template
```

Walks every schema property; for each one carrying `x-git-issue-type`
(`input`, `textarea`, `checkboxes`, or `dropdown`) emits a form field
with `description` and `options` pulled from the regular schema fields.
Output lands in `.github/ISSUE_TEMPLATE/new_dataset.yml`.

## Requirements

- Python 3.13
- Conda environment `apertus-data` (single shared env, no per-script deps)
