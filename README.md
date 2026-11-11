# DuetER

DuetER is a research prototype for efficient dual-path encrypted retrieval in
retrieval-augmented generation. It maps dense semantic similarity and a
compressed BM25 proxy to a common Euclidean ranking interface, ranks candidates
inside independently keyed compartment-local coordinate systems (compartment
conditional approximate distance-preserving encryption, CCADPE), and performs
exact native-score reranking and query-adaptive fusion (DuetRank) at the
client. Independent path aliases remove direct cross-path identifier equality.

This repository contains the DuetER implementation, its evaluation and
ablation scripts, compact paper-result artifacts, and the commands needed to
reproduce the reported experiments. It deliberately excludes corpora, model
checkpoints, cached embeddings, generated indexes, owner keys, and encrypted
record stores. Comparison-execution scripts for external baselines are not
part of this release; baseline numbers reported in the paper are either
author-reported literature values or clearly scoped controls documented in
the artifacts.

## Security scope

CCADPE is an efficiency and structured leakage-reduction mechanism, not a
replacement for HE, MPC, PIR, or ORAM. The cloud observes keyed
compartment/posting accesses, compartment sizes, candidate and timing traces,
and coordinates and distance orders inside each accessed compartment.
Independent aliases remove direct cross-path identifier equality; they do not
hide access-pattern correlation. The attack scripts under
`experiments/core/` (`attack_*.py`, `run_security_attacks.py`,
`run_compartment_security.py`) and the artifacts in `results/paper/security/`
quantify this operating point.

The NumPy pseudorandom generator is used for deterministic experiments. It
models the published bounded-noise distribution but is not a production
cryptographic PRF. A deployment must use a CSPRNG and proper key derivation.
The loopback protocol harness uses real AES-GCM wrappers but is a research
harness, not a production isolation or security evaluation.

## Repository layout

```text
experiments/
  core/                 Retrieval core: dual-path CCADPE implementation,
                        residual-IVF semantic index, keyed budgeted lexical
                        planner, DuetRank fusion, attacks, and ablations
  unified/              Unified complete-corpus quality pipeline (fixed seeds,
                        shared holdouts, three query-noise repeats) and the
                        loopback two-process protocol timing harness
                        (unified/protocol/)
  components/           Matched-budget component ablation (late routing,
                        plaintext-path controls, planner/quota, no native
                        scoring, fixed RRF)
scripts/                Reproduction and release-validation drivers
tests/                  Smoke and core unit tests
results/paper/          Compact JSON/figure artifacts backing each table
                        and figure of the paper
docs/                   Extended documentation and the experiment map
```

`experiments/core/pisces_reimplementation.py` and
`experiments/core/experiment_pisces_union_fusion.py` remain only because the
shared retrieval-core module (`run_experiment.py`) and the DuetRank recall
audit (`evaluate_duetrank_recall.py`) import them as internal controls. They
are a labeled functional emulator and an evaluation-rule reproduction; no
baseline comparison claims in this release are produced by them.

## Installation

Python 3.11/3.12 was used for the release artifact.

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For neural encoders, install the PyTorch wheel matching the local CUDA driver
first. The reported GPU runs used PyTorch 2.12.1 with CUDA 12.6. CPU-only
execution is supported but is not expected to reproduce GPU latency.

## Quick validation

The following checks do not download a dataset:

```bash
python scripts/check_release.py
python scripts/reproduce.py --profile quick --dry-run
```

`tests/test_smoke.py` and `tests/test_core.py` exercise the retrieval core
without external corpora; they require the dependencies in
`requirements.txt`.

## Paper reproduction

The reproduction driver prints commands unless `--execute` is supplied. This
makes the exact workload visible before a potentially long or
storage-intensive run.

```bash
python scripts/reproduce.py --profile full --dry-run        # RQ1 complete corpora
python scripts/reproduce.py --profile security --dry-run    # RQ3 attack suite
python scripts/reproduce.py --profile ablations --dry-run   # Tables IV-VI
python scripts/reproduce.py --profile components --dry-run  # Table VII
python scripts/reproduce.py --profile efficiency --dry-run  # protocol timing
```

The full corpora contain 2.68M NQ, 5.23M HotpotQA, and 8.84M MS MARCO
documents. Complete reproduction requires substantial disk space for source
corpora, embeddings, compartment coordinates, and indexes, and downloads are
explicit steps (`experiments/core/download_beir.py` verifies published MD5
checksums). The released result JSONs record the paper values without
redistributing those intermediates. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for dataset splits, main parameters,
hardware, seeds, and the mapping from paper claims to artifacts, and
[docs/EXPERIMENT_MAP.md](docs/EXPERIMENT_MAP.md) for a per-table/per-figure
guide from paper claims to code entry points.

## Data and models

`experiments/core/download_beir.py` downloads BEIR archives and verifies
their MD5 checksums. Dataset licenses and terms remain those of the original
providers. Neural runs expect a locally available
sentence-transformers-compatible model; the evaluated primary encoder is
`ibm-granite/granite-embedding-small-english-r2` (384 dimensions). No private
or personal data are included in this repository.

## License

The DuetER code in this repository is released under the MIT License. Dataset,
model, and third-party baseline licenses are not changed by this release.
