# Reproducibility guide

## Evaluated workloads

| Dataset | Corpus | Evaluated split |
|---|---:|---:|
| MS MARCO | 8,841,823 documents | 6,980 dev queries (43-query smoke run also retained) |
| NQ | 2,681,468 documents | 3,452 test queries |
| HotpotQA | 5,233,329 documents | 7,405 test queries |

Auxiliary corpora (20 Newsgroups, SciFact, NFCorpus) support the attack and
control experiments listed in `docs/EXPERIMENT_MAP.md`.

The main machine used an Intel Core i9-14900HX CPU and an NVIDIA GeForce RTX
4060 Laptop GPU. Latency should be interpreted with the execution mode and
hardware recorded in each result artifact.

## Main operating point

- semantic index: 2,048 mean-residual spherical-IVF compartments, two
  document assignments, 128 probes;
- semantic compartment projection dimension: 256 (released FP16 coordinates);
- semantic local return depth: 16/16/32 (NQ/HotpotQA/MS MARCO), client output
  depth 100;
- lexical representation: 1,024-D signed feature hash, 8,192-D DPE work
  space, project-then-lift to 32-D local projection, 16 compartments,
  per-partition local top-64, cap 300 into exact client BM25;
- lexical planner: HMAC-keyed SQLite FTS5 postings, cumulative posting budget
  50,000 (NQ/HotpotQA) or 250,000 (MS MARCO dev), candidate cap 1,000,
  planner order restored before the cap;
- CCADPE noise: scale 3.0, beta 0.10 (query radius 0.0375, database radius
  0.1125, uniform Euclidean ball);
- fusion: client-side DuetRank (HistGradientBoosting) with semantic backoff,
  calibrated only on the deterministic calibration half
  (`SHA256('outer:'+query_id)` split), with calibrated fixed-RRF and standard
  RRF controls; frozen across three query-noise repeats;
- complete-corpus seed: 20260917; query-noise repeat stride 100003.

The authoritative parameter file for the unified quality workload is
`experiments/unified/config.json`. Machine-specific paths in the packaged
configs, logs, and artifact provenance fields have been replaced by the
placeholders `<WORKSPACE>`, `<HOME>`, and `<KIMI_RUNTIME>`; point them at the
local preparation directories before executing reproduction profiles.

## Artifact-to-claim map

| Paper component | Released artifact | Code entry point |
|---|---|---|
| Table II: complete-corpus selectivity, fidelity | `results/paper/main/quality_comparison.json`, `main/dual_compartment_full_summary.json`, `ablations/semantic_depth_sweep/*_results.json` | `experiments/unified/unified_pipeline.py`, `experiments/core/benchmark_dual_compartment_full.py` |
| Figure 2: complete-corpus path utility (float-matched plain/protected) | `results/paper/main/figure2_path_utility_data.json`, `main/figure2_paired_intervals.json`, `main/figure2_complete_corpus_path_utility.png` | Protected side: `experiments/unified/unified_pipeline.py` + `experiments/core/benchmark_dual_compartment_full.py`; figure-assembly and plain-control rerun scripts are excluded from this release (data artifacts retained) |
| Dual-path fusion (DuetRank vs fixed/standard RRF) | `results/paper/main/quality_comparison.json`, `main/final_full_summary.json` | `experiments/core/experiment_candidate_rank_fusion.py`, `experiment_fixed_rrf_weight_sweep.py` |
| Figure 3 / online protocol cost (own system) | `results/paper/efficiency/protocol_timing_summary.json` | `experiments/unified/protocol/` |
| Table IV: semantic local-depth and probe-count sensitivity | `results/paper/ablations/nprobe*.json`, `ablations/semantic_depth_sweep/*_results.json` (`rq1_utility.semantic_sweep`) | `experiments/core/ablate_compartment_nprobe.py`, `benchmark_dual_compartment_full.py` |
| Table V: DuetRank calibration robustness | `results/paper/ablations/table5_duetrank_calibration_robustness.json` | `experiments/core/experiment_duetrank_calibration_robustness.py`, `ablate_duetrank_calibration_final.py` |
| Table VI: lexical selection (planner vs direct-300) | `results/paper/main/quality_comparison.json` (`direct300` methods), `main/lexical_dimension_decision.json` | `experiments/unified/run_direct_bm25.py`, `run_direct_fusion.py` |
| Table VII: matched-budget component effects | `results/paper/components/table7_paired_results.json`, `components/formal_v1/` | `experiments/components/` |
| Figure 4: leakage attacks (plaintext / global DPE / CCADPE) | `results/paper/security/` | `experiments/core/attack_*.py`, `run_security_attacks.py`, `run_compartment_security.py` |

## Reproduction profiles

`scripts/reproduce.py` defines the release profiles. Run with `--dry-run`
first.

- `quick`: dataset-free release validation and smoke tests;
- `full`: complete-corpus CCADPE quality runs for NQ, HotpotQA, and MS MARCO;
- `efficiency`: the loopback protocol timing harness (own system);
- `security`: the concrete-view attack suite;
- `ablations`: probe, local-depth, calibration, and lexical-selection
  sensitivity;
- `components`: the matched-budget component ablation;
- `all`: all of the above.

Full-corpus runs deliberately do not auto-download or silently build hundreds
of gigabytes of intermediates. Downloading and preprocessing are explicit
steps, and generated outputs go under ignored `results/generated/` or
`experiments/results/` directories.

## Determinism

All released programs expose or fix seeds; the unified pipeline derives
per-query noise seeds as `SHA256(repeat_seed:path:query_id)` from the master
seed (see `experiments/unified/config.json`). Exact retrieval metrics are
reproducible from the same corpus revision, split, model snapshot, cached
embeddings, and package versions. Timing is not deterministic and should be
remeasured after warm-up on the target hardware. GPU kernels and library
versions can introduce small numerical differences near ranking ties; the
timing harness's fresh-encoder diagnostics quantify this drift
(`results/paper/efficiency/protocol_timing_summary.json`).

Comparison-execution scripts for external baselines (Pisces, PRAG, ppRAG) are
intentionally excluded from this release; the paper reports those comparisons
with their scope and provenance notes.
