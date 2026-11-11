# DuetDPE experiments

This directory contains a reproducible implementation of the paper's retrieval core on BEIR SciFact and NFCorpus, keyed-index scaling runs on FiQA and TREC-COVID, million-document lexical-DPE stress tests, and complete-corpus MS MARCO, NQ, and HotpotQA semantic-DPE and dual-path experiments.

## What is implemented

- plaintext BM25;
- plaintext dense retrieval with an offline LSA-256 backend, `all-MiniLM-L6-v2`, or the primary `granite-embedding-small-english-r2` encoder;
- plaintext BM25+dense reciprocal-rank fusion;
- plaintext signed feature-hashed BM25, to isolate sketching loss;
- full-scan semantic and compressed BM25-to-L2 conditional DPE with client reranking;
- full-scan dual-path DPE, which isolates encryption loss from indexing loss;
- keyed Gaussian p-stable Euclidean multi-probe LSH versions of both paths and their dual-path fusion;
- a redesigned keyed residual spherical IVF semantic index: mean-centering removes the Granite common direction, spherical k-means creates 256 cells, each document is assigned to four cells, and each query probes 32 HMAC-labeled cells before DPE refinement;
- a streaming million-document realization using 2,048 cells, two assignments,
  a fixed 32/64/128-probe sweep, stored float16 conditional-DPE coordinates,
  server top-200 refinement, exact client top-100 reranking, and five complete
  query-noise repeats;
- a million-document lexical-DPE realization using unordered rare HMAC-posting
  candidates, a 1,024-dimensional signed-HMAC BM25 sketch, MIPS-to-L2 lifting,
  2,048-dimensional stored conditional-DPE coordinates, server top-200, exact
  client BM25 top-100 reranking, and five independent query-noise repeats;
- a cumulative-document-frequency-budget HMAC-FTS5 planner that replaces fixed
  two-term probing, evaluates caps 200/500/1000, caches raw rankings, and uses a
  deterministic NQ calibration/holdout split for the shared fusion setting;
- integrated stored lexical-DPE refinement over the budget-50K/cap-1000
  candidates, plus DuetRank: a target-domain calibrated, client-only candidate
  ranker with nested query splits and paired-bootstrap evaluation;
- a collision-free keyed BM25 tail index that omits the query-orthogonal residual coordinate, uses HMAC vocabulary postings with quantized BM25 weights, and retains encrypted L2 refinement;
- DuetDPE with independent path construction, query perturbation, residual-IVF semantic candidates, keyed lexical candidates, DPE refinement, and local fusion;
- an independently written Pisces functional reimplementation covering 32-bit SimHash coarse filtering, Hamming-radius expansion, multi-instance labeled PSI functionality, fixed-point shared BM25, and dual-path retrieval;
- direct shared-ID linkage and cross-path co-occurrence linkage attacks;
- a known-query least-squares plus orthogonal-Procrustes recovery attack against
  the executed padded semantic DPE transform;
- a 20-class auxiliary-corpus topic-inference attack against server-visible DPE
  coordinates;
- known-record expansion by scaled distance fingerprints and global assignment;
- ciphertext-neighborhood reconstruction measured by top-10 recall, ARI, and
  NMI;
- a five-repeat Gaussian query-noise utility sweep including formally calibrated
  `epsilon=1` and `epsilon=0.5`, `delta=1e-5` points;
- residual encrypted-neighborhood analysis;
- utility, local latency, communication, and offline setup measurements.

The NumPy random generator is used for reproducibility. It models the published bounded-noise distribution but is not a production cryptographic PRF.

## Dataset

The verified datasets are in `data/{scifact,nfcorpus,fiqa,trec-covid}`. Archives come from the official BEIR server:

```text
https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
MD5: 5f7d1de60b170fc8027bb7898e2efca1
nfcorpus MD5: a89dba18a62ef92f7d323ec890a0d38d
fiqa MD5: 17918ed23cd04fb15047f73e6c3bd9d9
trec-covid MD5: ce62140cb23feb9becf6270d0d1fe6d1
msmarco MD5: 444067daf65d982533ea17ebd59501e4
nq MD5: d4d3d2e48787a744b6f6e691ff534307
hotpotqa MD5: f412724f78b0d91183a0e86805e16114
```

## Run

From PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\.deps"
& 'python' .\download_beir.py nfcorpus fiqa trec-covid
$env:PYTHONPATH = "$PWD\.deps"
& 'python' .\run_experiment.py --data-dir .\data\scifact --dataset-name scifact --dense-backend sentence-transformer --model .\cache\models\all-MiniLM-L6-v2 --results-dir .\results\scifact_minilm_v2
& 'python' .\benchmark_keyed_bm25_index.py --dataset trec-covid
& 'python' .\benchmark_keyed_fts5_scale.py --dataset msmarco --max-docs 1000000
& 'python' .\benchmark_keyed_fts5_scale.py --dataset msmarco --max-docs 1000000 --max-probe-terms 2 --reuse-index
& 'python' .\benchmark_semantic_candidate_index.py --dataset scifact --output .\results\semantic_index_scifact.json
& 'python' .\benchmark_million_semantic_hybrid.py --dataset msmarco --max-docs 1000000 --device cuda
& 'python' .\summarize_million_semantic_hybrid.py
& 'python' .\benchmark_million_lexical_dpe.py --dataset msmarco --max-docs 1000000 --device cuda
& 'python' .\summarize_million_lexical_dpe.py
& 'python' .\benchmark_budgeted_lexical_candidates.py --dataset nq --scale million --posting-budgets 10000 50000 250000 --caps 200 500 1000 --max-cap 1000 --output .\results\budgeted_lexical\nq_million.json
& 'python' .\summarize_budgeted_lexical_validation.py
& 'python' .\benchmark_budgeted_lexical_dpe.py --dataset nq --cloud-depth 200 --repeats 5 --output .\results\budgeted_lexical_dpe\nq_d200.json
& 'python' .\experiment_candidate_rank_fusion.py --datasets nq --lexical-source budgeted-dpe --output .\results\budgeted_lexical_dpe\learned_fusion_nq.json
& 'python' .\summarize_duetrank_results.py
& 'python' .\benchmark_msmarco_dev_batched.py --repeats 5 --output-dir .\results\msmarco_dev_full_semantic
& 'python' .\benchmark_budgeted_lexical_candidates.py --dataset msmarco --scale full --split dev --semantic-rankings .\results\msmarco_dev_full_semantic\rankings.npz --posting-budgets 50000 250000 --caps 200 500 1000 --max-cap 1000 --primary-budget 50000 --output .\results\budgeted_lexical\msmarco_dev_full.json
& 'python' .\benchmark_budgeted_lexical_dpe.py --dataset msmarco --scale full --split dev --budget-key 250000 --candidate-rankings .\results\budgeted_lexical\msmarco_dev_full.raw.npz --semantic-rankings .\results\msmarco_dev_full_semantic\rankings.npz --cache-tag msmarco_dev_b250k --cloud-depth 500 --output-depth 300 --repeats 5 --output .\results\budgeted_lexical_dpe\msmarco_dev_full_b250k_d500_o300.json
& 'python' .\summarize_full_corpus_closure.py
& 'python' .\benchmark_million_semantic_hybrid.py --dataset msmarco --max-docs 8841823 --cache-root .\cache\full_semantic --results-root .\results\full_semantic_hybrid --device cuda --batch-size 64 --encoder-chunk 8192 --dense-query-batch 128 --dense-document-block 65536 --repeats 5 --skip-ivf-exact
& 'python' .\summarize_full_corpus_semantic_hybrid.py
& 'python' .\calibrate_feature_hash.py
& 'python' .\analyze_known_pair_recovery.py
& 'python' .\run_security_attacks.py
& 'python' .\evaluate_dp_utility.py
& 'python' .\run_ablations.py
```

The default runner uses TF-IDF followed by TruncatedSVD(256). The primary audited neural runs use the local Granite snapshot on an RTX 4060 Laptop GPU; MiniLM is retained as an encoder-sensitivity experiment. CUDA PyTorch is isolated under `.gpu_deps` and activated with `DUETDPE_GPU_DEPS=1`; CPU reproducibility remains available through `.deps`. Granite embeddings are cached under `cache/granite_gpu`.

```powershell
$env:DUETDPE_GPU_DEPS = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& 'python' .\run_experiment.py --data-dir .\data\scifact --dataset-name scifact --cache-dir .\cache\granite_gpu --results-dir .\results\scifact_granite_gpu --dense-backend sentence-transformer --model .\cache\models\granite-embedding-small-english-r2 --batch-size 64 --device cuda
```

Outputs:

- `results/scifact_granite_residual_ivf/results.json` and `results/nfcorpus_granite_residual_ivf/results.json`: primary Granite runs with the redesigned semantic index;
- `results/semantic_index_scifact.json` and `results/semantic_index_nfcorpus.json`: 24-configuration residual-IVF sweeps;
- `results/feature_hash_calibration.json`: train-split analytic and fixed-seed signed-hash error calibration for 512/1024/2048 dimensions;
- `results/known_pair_recovery.json`: actual-Granite known-query transform and
  static-document reconstruction stress test at the deployed DPE parameters;
- `results/security_attacks.json`: topic inference, known-record expansion,
  neighborhood reconstruction, and the cross-view linkage result provenance;
- `results/dp_utility.json`: five-repeat semantic Gaussian-noise utility sweep;
- `results/scifact_minilm_v2/results.json` and `results/nfcorpus_minilm_v2/results.json`: MiniLM sensitivity results;
- `results/final/retrieval_metrics.csv`: paper-ready retrieval/latency table;
- `results/final/comparisons.csv`: paired utility-retention and overhead comparisons;
- `results/final/REPORT.md`: concise human-readable report;
- `results/lexical_index_analysis.json`: MIPS residual-coordinate distribution and candidate-cap sweep;
- `results/keyed_bm25/*.json`: five-repeat collision-free keyed-index scale and cap sweeps;
- `results/keyed_fts5/*.json`: disk-backed HMAC-term/FTS5 scale runs. This is an
  exact-BM25 candidate implementation, not the in-memory 16-bit quantized
  implementation; subset policies are recorded in every result;
- `results/budgeted_lexical/validation_summary.json`: fixed-operating-point
  coverage, latency, semantic/hybrid utility, and deterministic NQ holdout
  validation for the redesigned lexical planner;
- `results/budgeted_lexical_dpe/FINAL_SUMMARY.json`: final stored lexical-DPE,
  calibrated fixed-RRF, DuetRank held-out/bootstrap, client model size/latency,
  and negative-transfer controls;
- `results/budgeted_lexical_dpe/FINAL_FULL_SUMMARY.json`: complete-corpus
  candidate-scoped lexical-DPE, outer-holdout DuetRank, official MS MARCO dev,
  paired confidence intervals, and the explicit offline-storage scope warning;
- `results/budgeted_lexical_dpe/fixed_rrf_weight_sweep.json`: complete-corpus
  0.00--1.00 semantic-weight sweep at 0.01 resolution, calibration-only weight
  selection, outer-holdout metrics, and direct measurements of lexical Top-10
  promotion, lexical-exclusive admission, and relevant-document rescue;
- `results/budgeted_lexical_dpe/pisces_union_fusion.json`: exact reproduction
  of the Pisces artifact's unordered semantic@k-union-lexical@k evaluation on
  the three complete corpora, including equal-output-budget semantic@2k
  controls, lexical rescue rates, and paired bootstrap intervals;
- `results/budgeted_lexical_dpe/duetrank_recall_audit.json`: five-repeat
  outer-holdout Recall@10/Recall@20 and hit-rate audit for the final DuetRank
  outputs, matched-maximum and matched-actual-cardinality comparisons with the
  Pisces Top-10 union, paired intervals, and exact nDCG consistency checks;
- `results/budgeted_lexical_dpe/duetrank_{nq,hotpotqa,msmarco}_outer.rankings.npz`:
  cached five-repeat final DuetRank rankings and deterministic outer masks used
  by the Recall@10/@20 audit;
- `results/msmarco_dev_full_semantic/`: five-repeat full-corpus semantic DPE over
  all 6,980 official dev queries using the exactly audited cell-major evaluator;
- `results/ablations/scifact_granite.json`: primary Granite DPE-margin, quantization, cap, and dummy-access ablations;
- `results/pisces_filter_validation.json`: corrected Pisces coarse-filter validation.
- `results/million_semantic_hybrid/{msmarco,nq,hotpotqa}_1000000/`: complete
  JSON results and compressed rankings for the three million-document semantic
  and hybrid runs;
- `results/million_semantic_hybrid/{REPORT.md,summary.csv,manifest.json}`:
  consolidated results, validated ranking shapes, and SHA-256 provenance.
- `results/million_lexical_dpe/{msmarco,nq,hotpotqa}_1000000_h1024/`:
  measured lexical-DPE results and compressed rankings; the directory-level
  `REPORT.md`, `summary.csv`, and `manifest.json` validate and consolidate them;
- `results/full_semantic_hybrid/{msmarco_8841823,nq_2681468,hotpotqa_5233329}/`:
  complete-corpus JSON results and compressed five-repeat rankings;
- `results/full_semantic_hybrid/{REPORT.md,summary.csv,manifest.json}`:
  consolidated complete-corpus results, validated array shapes, and SHA-256
  provenance;
- `results/official_pisces_vm/official_pisces_summary.{md,csv,json}`: official
  Pisces BM25 and semantic cryptographic execution on 256- and 1,990-context
  CLAPNQ subsets; the evidence directory retains logs, hashes, environment, and
  compiler-compatibility patches;
- `results/official_pisces_vm/official_unit_tests_20260825.md`: official Pisces
  BM25 and similarity GoogleTest reconstruction, pass counts, resource use,
  hashes, and links to the retained XML/log evidence.

The final report keeps legacy LSH as a control and reports residual-IVF candidate count, top-100 coverage, relevant-document coverage, posting reads, storage, and downstream DPE utility. HMAC cell labels conceal centroid identifiers but still leak label equality, posting sizes, query access patterns, and candidate overlap. The manual Pisces implementation is a functional emulator: HMAC/direct dictionaries and local arithmetic model ideal protocol outputs, but do not instantiate blind OPRF, oblivious OKVS, additive HE, garbled circuits, or PIR. Its Python latency is not comparable to official Pisces cryptographic latency. See `PISCES_VALIDATION.md`.

## Baseline policy

The repository never labels a simplified surrogate as an official external
protocol. The official Pisces source is retained under
`baselines/pisces_official_reference`; official binaries were built and run in
the Ubuntu VMware guest, while the Python implementation remains labeled
`*_reimpl`. ppRAG/CAPRISE is marked as a behavioral reimplementation. The 2026
PRAG paper's reported measurements are retained as cited literature values;
its newly identified official repository has not been locally executed.
`OFFICIAL_BASELINE_STATUS.md` records revisions, patches, execution
topology, results, and the distinction between 2026 PRAG and the unrelated 2024
MPC-PRAG repository.
