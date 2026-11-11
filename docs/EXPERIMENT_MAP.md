# Experiment map: from paper claims to code

This guide walks each table and figure of the paper back to its code entry
point, configuration, and compact result artifact. All paths are relative to
the repository root. Every artifact records its own seeds, configuration
hashes, and scope notes; where a result has a documented limitation, the
limitation text is embedded in the artifact itself.

## RQ1 — Complete-corpus utility and scalability

### Table II — candidate selectivity and retrieval fidelity

- Code: `experiments/core/benchmark_dual_compartment_full.py` (dual-path
  CCADPE complete-corpus runner), driven and unified by
  `experiments/unified/unified_pipeline.py` with `experiments/unified/config.json`.
- Entry stages: `experiments/unified/run_quality_stage.py --stage semantic`
  (plus `run_semantic_cloud.py` / `run_semantic_rerank_io.py`),
  `--stage lexical`, `--stage fusion`.
- Artifacts: `results/paper/main/quality_comparison.json` (held-out fused
  quality and paired comparisons), `main/dual_compartment_full_summary.json`
  (scan fraction, relevant-candidate coverage, reference-vs-CCADPE nDCG),
  `ablations/semantic_depth_sweep/*_results.json` (per-dataset detail,
  including the `rq1_utility.semantic_sweep` block).

### Figure 2 — complete-corpus path utility

- The Protected (CCADPE) side is reproduced by the unified pipeline and core
  dual-path runner above. The Plain float-matched control recomputed native
  FP32 scoring over the same routing, planner pools, and frozen fusion models;
  its rerun/assembly scripts and the plotting script are excluded from this
  release (they depend on historical study directories), and the frozen data
  artifacts are retained instead.
- Artifacts: `results/paper/main/figure2_path_utility_data.json`,
  `main/figure2_paired_intervals.json`,
  `main/figure2_complete_corpus_path_utility.png`.

### DuetRank fusion and controls

- Code: `experiments/core/experiment_candidate_rank_fusion.py` (DuetRank
  training/evaluation with nested deterministic splits),
  `experiment_fixed_rrf_weight_sweep.py` (101-point calibration-only weight
  sweep), `experiment_pisces_union_fusion.py` (Pisces-artifact union rule),
  `evaluate_duetrank_recall.py` (Recall@10/@20 audit).
- Artifact: `results/paper/main/final_full_summary.json`.

## RQ2 — Efficiency

### Online protocol cost (Figure 3, own system)

Loopback two-process protocol timing (DuetER and its direct-BM25 controls):
`experiments/unified/protocol/` (`timing_suite.py`, `server.py`, `client.py`,
`wire.py`, `records.py`, `validate_suite.py`, `postprocess_timing.py`).
Architecture and scope:
`experiments/unified/protocol/ARCHITECTURE_AND_RUN.txt`.
Artifact: `results/paper/efficiency/protocol_timing_summary.json`.

The paper additionally situates this cost against Pisces, PRAG, and
ppRAG-style executions; those baseline comparison scripts and artifacts are
not part of this release (see README "Scope").

## RQ3 — Empirical leakage validation

### Figure 4 — attacks from plaintext, global-DPE, and CCADPE views

- Code (all under `experiments/core/`):
  - cross-view identifier linkage: `attack_cross_path_anchor_fingerprints.py`,
    `attack_cross_path_cooccurrence.py`, `attack_cross_path_posting_access.py`;
  - compartment stitching: `attack_cross_compartment_stitching.py`,
    `attack_lexical_stitching_d8192_rc32.py` (+ `_b_sweep`),
    `attack_lexical_stitching_rc32.py`;
  - known-query / known-record recovery:
    `analyze_known_pair_recovery.py`, `calibrate_feature_hash.py`;
  - parameter-mitigation sweep: `attack_semantic_parameter_mitigation.py`;
  - drivers: `run_security_attacks.py`, `run_compartment_security.py`.
- Artifacts: `results/paper/security/` (cross-path linkage, stitching,
  known-pair recovery, mitigation sweep).

## Ablation studies

### Table IV — semantic parameter sensitivity

- Probe count: `experiments/core/ablate_compartment_nprobe.py` →
  `results/paper/ablations/nprobe*.json`.
- Local return depth: semantic sweep recorded in
  `results/paper/ablations/semantic_depth_sweep/*_results.json`
  (`rq1_utility.semantic_sweep`); lexical depth:
  `ablate_compartment_larger_lexical_depth.py` →
  `ablations/larger_lexical_depth.json`.

### Table V — DuetRank calibration robustness

- Code: `experiments/core/experiment_duetrank_calibration_robustness.py`,
  `ablate_duetrank_calibration_final.py` (ten deterministic seeds, nested
  calibration subsets).
- Artifact: `results/paper/ablations/table5_duetrank_calibration_robustness.json`.

### Table VI — lexical dimension and selection

- Code: `experiments/unified/run_direct_bm25.py` + `run_direct_fusion.py`
  (direct-300/direct-1000 controls sharing the planner and native scorer);
  lexical width decision:
  `experiments/unified/lexical_decision/build_decision_report.py`.
- Artifacts: `results/paper/main/quality_comparison.json` (`direct300`
  methods), `main/lexical_dimension_decision.json`.

### Table VII — matched-budget component effects

- Code: `experiments/components/` (`run_components.py`,
  `prepare_components.py`, `server_adapter.py`, `components_client.py`,
  `proxy_entry.py`, `build_plain_index.py`, `native_cache_bridge.py`,
  `summarise_components.py`, `validate_components.py`; quality stage under
  `components/quality/`). The registered design, including the common
  candidate-admission rule `B*`, is in `EXPERIMENT_PLAN.txt`.
- Artifacts: `results/paper/components/table7_paired_results.json`,
  `components/formal_v1/`.

## Tests

`tests/test_smoke.py` and `tests/test_core.py` run dataset-free checks over
keyed labels, residual IVF, the keyed BM25 index, padded lexical-DPE MIPS
order, budget enforcement, candidate-scoped materialization, DuetRank
missing-path handling, and the Pisces functional paths.
