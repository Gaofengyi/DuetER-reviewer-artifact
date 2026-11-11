# Experiment completion status

## Completed and audited

- Official BEIR archives downloaded, safely extracted, and matched to the published MD5 values: SciFact, NFCorpus, FiQA, and TREC-COVID.
- Local neural model snapshots: `sentence-transformers/all-MiniLM-L6-v2` and `ibm-granite/granite-embedding-small-english-r2`.
- Neural retrieval-core runs with Granite and MiniLM on SciFact and NFCorpus, including plaintext, full-scan DPE, legacy indexes, the redesigned keyed residual spherical IVF semantic index, collision-free keyed BM25 tail indexing, query cover probes, and the independently written Pisces functional emulator.
- Granite GPU encoding on an RTX 4060 Laptop GPU: SciFact corpus 21.00 seconds and queries 0.30 seconds; NFCorpus corpus 16.76 seconds and queries 0.49 seconds.
- Five-repeat keyed-index cap sweeps on SciFact, NFCorpus, FiQA, and TREC-COVID (171,332 documents).
- SciFact DPE-margin, 8/12/16-bit quantization, cap, and 0/2/4/8/16 dummy-token access-padding ablations.
- Train-split signed feature-hash calibration at 512/1024/2048 dimensions on SciFact and NFCorpus, including the exact analytic variance and fixed-seed collision errors.
- Known-query Procrustes recovery using 623 actual Granite query embeddings and
  1,000 static SciFact document embeddings at d=384, padded D=512, beta=0.1,
  scale=3, with one/four/sixteen repeated releases and sigma in {0, 0.015}.
- Four concrete attack families: cross-view linkage, 20-class topic inference
  on 4,500 20 Newsgroups texts, distance-fingerprint known-record expansion,
  and ciphertext neighborhood reconstruction.
- Five-repeat semantic Gaussian-noise utility sweeps on SciFact and NFCorpus,
  including the formally calibrated sigma=9.69 (epsilon=1, delta=1e-5) and
  sigma=19.38 (epsilon=0.5, delta=1e-5) settings.
- Official BEIR MS MARCO, NQ, and HotpotQA archives downloaded, extracted, and
  verified against their published MD5 values; the same disk-backed HMAC-term
  index was executed on declared one-million-document relevance-preserving
  subsets of all three corpora.
- Million-document Granite semantic and dual-path runs on the same three
  subsets: streaming embeddings, 2,048-cell two-assignment residual IVF,
  HMAC cell labels, a fixed 32/64/128-probe sweep, stored conditional-DPE
  coordinates, exact client reranking, HMAC-FTS5 lexical retrieval, RRF, and
  five complete DPE query-noise repeats.
- Million-document lexical-DPE runs on all three corpora: unordered two-rare-term
  HMAC-posting candidates, 1,024-dimensional signed-HMAC BM25 sketches,
  MIPS-to-L2 lifting, stored padded 2,048-dimensional DPE coordinates, server
  top-200, exact client BM25 top-100 reranking, and five DPE query-noise repeats.
- Complete-corpus semantic and hybrid executions over all 8,841,823 MS MARCO,
  2,681,468 NQ, and 5,233,329 HotpotQA records: streaming exact-dense reference,
  2,048-cell residual IVF, stored DPE coordinates, the independently built
  full-corpus HMAC-FTS5 lexical branch, client RRF, and five DPE repeats.
- Budget-aware lexical query planning over all three one-million-document
  indexes: cumulative document-frequency budgets of 10K/50K (plus 250K on MS
  MARCO and NQ), caps 200/500/1000, exact BM25 top-100, cached rankings, and
  semantic-heavy RRF. The shared 50K/cap-1000/weight-.98 point is selected on a
  deterministic half of NQ and checked on the other half and both transfer
  datasets.
- Stored lexical-DPE refinement over the new 50K/cap-1000 candidate sets on all
  three million-document subsets: cloud top-200, exact client BM25 top-100, and
  five paired lexical/semantic DPE repeats. Target-domain DuetRank uses nested
  deterministic query splits, paired-query bootstrap, saved client models, and
  calibrated fixed-RRF controls on NQ and HotpotQA.
- Complete-corpus budgeted lexical-DPE and DuetRank runs: every candidate row
  touched by the fixed full-index workloads is materialized before query
  execution, with 1.56M NQ, 2.73M HotpotQA, and 3.89M MS MARCO-dev coordinate
  rows, five DPE repeats, exact client BM25, nested calibration, and paired
  bootstrap. Deployment-wide offline coordinate storage is deliberately not
  counted by this workload-scoped optimization.
- Official 6,980-query MS MARCO dev execution over all 8,841,823 passages. A
  cell-major semantic-DPE evaluator is exactly checked against query-major
  execution on four queries, then used for five full dev repeats; 50K/250K
  lexical budget sweeps and actual 250K lexical DPE are also executed.
- Twenty-five unit tests, including keyed-label and identical-vector checks for residual IVF, the keyed BM25 index, padded lexical-DPE MIPS order and disk-query checks, the million-subset policy/RRF metrics, budget enforcement, candidate-scoped materialization, explicit MS MARCO dev loading, DuetRank missing-path/bootstrap checks, and the audited Pisces functional paths.

## Principal results

- SciFact Granite plaintext RRF: 0.7251 nDCG@10 / 0.9833 Recall@100.
- SciFact Granite residual-IVF dual DPE: 0.7090 / 0.9700; query-cover variant: 0.7128 / 0.9667.
- NFCorpus Granite plaintext RRF: 0.3603 / 0.3274; residual-IVF dual DPE: 0.3578 / 0.3338.
- With the shared K=256, four-assignment, 32-probe configuration, semantic candidates fall from 5,177.2 to 1,605.5 on SciFact and from 3,629.4 to 1,157.0 on NFCorpus. Semantic DPE retains 99.84% and 98.46% of full-scan nDCG@10, respectively.
- Pisces Granite semantic emulator retains 99.50% of plaintext dense SciFact nDCG@10, but its SimHash filter returns 4,513.5 of 5,183 documents on average; the redesigned index returns 1,605.5.
- At cap 200, positive-score BM25 top-100 coverage is 1.0000 on SciFact, FiQA, and TREC-COVID, and 0.9978 on NFCorpus.
- Five-run index candidate latency at cap 200: SciFact 0.195 +/- 0.006 ms, NFCorpus 0.045 +/- 0.003 ms, FiQA 6.176 +/- 0.048 ms, and TREC-COVID 19.265 +/- 0.130 ms.
- On SciFact with Granite, increasing beta from 0 to 0.20 changes dual full-scan nDCG@10 from 0.6981 to 0.6984, while semantic/lexical encrypted top-10 neighborhood overlap falls only to 0.942/0.912. This remains substantial structural leakage.
- Sixteen cover terms reduce mean candidate-set Jaccard against the unpadded query to 0.848 with 2.02% more posting reads and unchanged positive top-100 coverage. This is not a differential-privacy guarantee.
- With 623 exact known query embeddings, one unperturbed release recovers static
  document embeddings at 0.9982 mean cosine.  At heuristic sigma=0.015 this is
  0.9164 for one release and 0.9907 after averaging sixteen releases, confirming
  that the executed coordinate view is structurally vulnerable to known pairs.
- Topic inference reaches 0.6440 accuracy from DPE coordinates over 20 balanced
  classes (random 0.05; Wilson 95% interval 0.6194--0.6678), versus 0.6473 from
  plaintext coordinates. Heuristic sigma=0.015 lowers it only to 0.6167.
- Across ten samples, two known document anchors align 74.87% of 300 targets by
  ciphertext distance fingerprints; four anchors align 99.93%.
- Across ten 512-document samples, ciphertext geometry preserves 96.30% of
  plaintext top-10 neighbors, with mean ARI 0.6190 and NMI 0.7251 for 20
  clusters.
- Formal Gaussian noise under arbitrary-query adjacency is not practically
  useful: at sigma=9.69, SciFact/NFCorpus nDCG@10 falls to 0.0007/0.0096.
- The one-million-document disk index contains 57.27M token occurrences, builds
  in 58.04 seconds, and occupies 226.96 MB. All-term exact candidate generation
  is slow (1520.02 ms mean / 3192.68 ms p95). Two rare keyed probes reduce this
  to 6.90 / 23.60 ms but return only 64.56% of judged relevant documents in the
  200-candidate set, exposing a sharp latency--recall trade-off.
- At the same two-rarest-term point, the NQ subset builds in 281.29 seconds,
  occupies 314.62 MB, and gives 27.56/80.44 ms mean/p95 with 67.10% candidate
  recall over 3,452 queries. HotpotQA builds in 252.84 seconds, occupies 248.71
  MB, and gives 7.68/25.91 ms with 64.21% candidate recall over 7,405 queries.
- With the shared 50K posting budget and cap 1000, judged-relevant candidate
  coverage rises to 80.29%/89.62%/89.51% on MS MARCO/NQ/HotpotQA, absolute
  gains of 15.73/22.52/25.30 percentage points. Mean selected terms are
  3.12/4.13/7.48, mean estimated postings 23.33K/32.46K/36.87K, and measured
  mean/p95 latency 13.86/27.66, 20.33/28.82, and 24.10/31.76 ms.
- At the same shared semantic RRF weight 0.98, hybrid nDCG@10 is
  0.6086/0.5611/0.7212 versus semantic-only 0.6045/0.5598/0.7166. On the 1,706
  held-out NQ queries it is 0.56576 versus 0.56474, with 89.59% candidate
  coverage. No per-dataset weight is used for this claim.
- Under actual budgeted lexical-DPE, plaintext-sketch/DPE lexical nDCG@10 is
  0.33166/0.33168 on NQ and 0.69888/0.69885 on HotpotQA. On 1,739 and 3,740
  outer-held-out queries, DuetRank raises semantic nDCG@10 from 0.55167 to
  0.56066 and from 0.71969 to 0.77489. Absolute-gain paired-bootstrap 95% CIs
  are [0.00540,0.01250] and [0.05013,0.06040].
- Calibration-selected fixed RRF reaches 0.55337 (weight 0.98) on NQ and
  0.74393 (weight 0.80) on HotpotQA; DuetRank still adds 0.00729 and 0.03096.
  Client models are 265/239 KB and measured inference is 1.04/1.27 ms/query.
  Simple query routing and zero-shot NQ+HotpotQA-to-MS transfer both fail, so
  target-domain relevance calibration is required.
- On complete-corpus outer holdouts, actual lexical-DPE DuetRank raises NQ
  nDCG@10 from 0.51492 to 0.51910 (95% CI [0.00198,0.00639]), HotpotQA from
  0.64108 to 0.68695 ([0.04117,0.05078]), and MS MARCO dev from 0.28360 to
  0.30224 ([0.01301,0.02408]). Client inference is 1.60--1.75 ms/query.
- A leakage-neutral 101-point fixed-RRF sweep selects semantic weights
  0.98/0.85/0.85 on calibration-only NQ/HotpotQA/MS MARCO splits.  Outer-heldout
  nDCG@10 is 0.51253/0.65527/0.29304 versus semantic-only
  0.51492/0.64108/0.28360.  Thus fixed fusion is not robust on NQ, while the
  HotpotQA/MS gains are significant.  For RRF constant 60, any semantic weight
  above 70/131 cannot admit a lexical-only document into Top-10; all three
  selected points consequently have zero lexical-exclusive Top-10 documents.
  The lexical path only reorders semantic candidates through rank overlap.
- Reproducing the Pisces artifact's dual-path evaluation rule---the unordered
  set union of semantic@k and lexical@k---raises outer-heldout relevant recall
  at k=10 by 0.05808/0.08262/0.10848 on NQ/HotpotQA/MS MARCO, while returning
  17.28/17.56/17.53 documents on average.  Against the equal-maximum-budget
  semantic@20 control, the differences are -0.02755/+0.03984/-0.01045; only
  HotpotQA is significantly better, while the MS interval includes zero.  The
  Pisces union is therefore a recall-oriented candidate expansion, not an
  ordered Top-10 fusion and cannot itself produce nDCG@10.
- A five-repeat outer-holdout audit of the final, fixed-size DuetRank Top-10
  raises Recall@10 from 0.73265 to 0.73840 on NQ, 0.66644 to 0.72128 on
  HotpotQA, and 0.46757 to 0.50147 on MS MARCO.  Absolute gains are
  0.00575/0.05484/0.03390; the NQ two-sided 95% interval narrowly includes zero,
  while HotpotQA and MS MARCO are significant.  Recomputed nDCG matches every
  stored primary result exactly.
- Under the matched maximum final-output budget, DuetRank@20 reaches Recall@20
  0.82413/0.76305/0.61732 on NQ/HotpotQA/MS MARCO versus Pisces-style
  semantic@10-union-lexical@10 recall 0.79073/0.74906/0.57605.  Gains are
  0.03340/0.01398/0.04126 and all paired intervals exclude zero.  Matching the
  Pisces union's actual per-query cardinality (17.28/17.56/17.53 on average)
  leaves gains of 0.02401/0.00636/0.02510, also significant.  This is a matched
  final-evidence-budget system comparison, not an aggregation-only claim:
  DuetRank scores deeper semantic and lexical candidate pools and uses
  target-domain calibration, whereas Pisces unions only the two Top-10 sets.
- Complete-corpus lexical plaintext/DPE nDCG@10 is 0.27576/0.27560 on NQ,
  0.57852/0.57850 on HotpotQA, and 0.21987/0.21968 on MS MARCO dev. Steady
  lexical-DPE mean/p95 is 2.09/3.78, 7.92/16.18, and 10.71/17.74 ms/query.
- The 6,980-query MS MARCO semantic DPE run reaches 0.28810 nDCG@10 and 0.78190
  Recall@100. Cell-major five-repeat scoring plus exact reranking takes 349.81s
  (99.77 query-repeat/s); a four-query query-major audit matches every top-100
  ranking and set exactly. Raising the posting budget from 50K to 250K raises
  candidate coverage from 0.7036 to 0.8115 but planner mean/p95 from
  21.00/44.08 to 85.08/162.67 ms.
- The million-document semantic index selects 128 probes on all three corpora
  under the fixed dual-coverage rule and scans 9.18--9.91% of the corpus.
  Exact-dense top-100 coverage is 0.9705/0.9719/0.9641 and judged-relevant
  coverage is 0.9201/0.9853/0.9746 for MS MARCO/NQ/HotpotQA; MS MARCO therefore
  does not meet the 0.95 relevant-coverage target and uses the prespecified
  maximum rather than a passing setting.
- DPE semantic nDCG@10 / Recall@100 is 0.6045/0.6223 (MS MARCO),
  0.5598/0.9499 (NQ), and 0.7166/0.8388 (HotpotQA), retaining
  97.25--99.15% of exact-dense nDCG. Mean semantic online time across five
  complete repeats is 14.42--15.78 ms per query on the RTX 4060.
- Under the original two-rarest-term/cap-200 policy, DPE-plus-keyed-lexical RRF nDCG@10 / Recall@100 is 0.5634/0.6765,
  0.3943/0.9493, and 0.6308/0.8454, respectively. Relative to exact dense plus
  the same lexical path, fusion retains 98.12--99.98% nDCG. The fast lexical
  path can reduce nDCG relative to dense-only, so the result validates executed
  composition and scale rather than a universal hybrid-quality gain. The later
  budget-aware planner, stored lexical-DPE refinement, and DuetRank repair this
  failure on held-out NQ and HotpotQA queries.
- On the three million-document lexical-DPE runs, mean candidates are
  7,575.6/6,220.2/1,194.4 for MS MARCO/NQ/HotpotQA. Relative to plaintext
  signed-hash MIPS on exactly the same candidates and the same client reranker,
  DPE retains 100.00%/100.00%/100.01% nDCG@10 and
  99.61%/99.91%/99.98% Recall@100. Mean measured online time is
  13.13/6.44/3.49 ms, respectively. These conditional retention values do not
  certify global BM25 ordering: relevant-document candidate coverage is only
  0.8432/0.9059/0.7576.
- On the complete corpora, the selected 128 probes scan 9.08--9.73% of vectors.
  Exact-dense/DPE-semantic nDCG@10 is 0.5201/0.5035 (MS MARCO),
  0.5212/0.5178 (NQ), and 0.6468/0.6398 (HotpotQA), corresponding to
  96.82%, 99.36%, and 98.92% retention. Judged-relevant candidate coverage is
  0.9128/0.9876/0.9762; MS MARCO still misses the 0.95 gate.
- Under the original two-rarest-term policy, complete-corpus DPE-hybrid
  nDCG@10/Recall@100 is 0.4700/0.5115,
  0.3489/0.9206, and 0.5185/0.7781. The HMAC-FTS5 branch again lowers nDCG
  relative to dense-only; the later budgeted DPE/DuetRank runs above replace
  rather than conceal this negative ablation.
- Out-of-core semantic online mean/p95 is 603.63/2274.48 ms (MS MARCO),
  33.91/45.16 ms (NQ), and 67.58/83.19 ms (HotpotQA). MS MARCO's 43-query run
  exposes cold/random memmap materialization and must not be compared directly
  with the earlier GPU-resident million-vector timings.

## Explicitly incomplete

- The released Pisces C++ cryptographic system was executed in the Ubuntu
  VMware guest on its released CLAPNQ-scale benchmark, but not at one million
  documents. The Python result remains a functional emulator and is not used as
  cryptographic latency.
- The 2026 PRAG v2 paper now identifies its official C++/Microsoft SEAL 4.1
  repository. Its 100K-TriviaQA results are included in the manuscript only as
  author-reported literature values, with workload and hardware disclosed; no
  local matched-workload PRAG execution is claimed. The 2024 CrypTen/MPC-PRAG
  repository remains a different system.
- No end-to-end generator/QA evaluation; million-document official cryptographic baseline;
  or isolated five-run end-to-end systems benchmark has been completed. The
  subset and complete-corpus retrieval results are reported separately. The
  complete-corpus lexical-DPE query workloads materialize every accessed row,
  but do not measure deployment-wide coordinate precomputation/storage. The
  new budget-aware planner does not constitute a specialized Block-Max WAND
  implementation.
- A LaTeX engine is not installed in the environment, so the paper source has balanced braces and updated tables but no compiled-PDF layout verification.

These omissions are deliberately left visible; no surrogate values are labeled as official baseline or production cryptographic measurements.
