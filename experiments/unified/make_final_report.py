"""Assemble final tables only after every required quality and timing run exists."""
from pathlib import Path
import json, sys, hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
C=read(HERE/'config.json');Q=read(HERE/'quality_comparison.json')
PROTOCOL=read(HERE/'protocol'/'PROTOCOL_TIMING_SUMMARY.json')
assert PROTOCOL['status']=='complete' and PROTOCOL['invariant_checks']['formal_runs']==9
DS=list(C['datasets']);names={'nq':'NQ','hotpotqa':'HotpotQA','msmarco':'MS MARCO'}
def paired(base,system,mask):
    d=(system[:,mask]-base[:,mask]).mean(axis=0)
    rng=np.random.default_rng(C['fusion']['bootstrap_seed']);v=[]
    for _ in range(20):v.extend(d[rng.integers(0,len(d),(250,len(d)))].mean(axis=1))
    return {'mean_difference':float(d.mean()),'ci95':np.percentile(v,[2.5,97.5]).tolist()}
P={};paired_pprag={};timing={};ptiming={}
for ds in DS:
    cfg=C['datasets'][ds];base=HERE/'pprag'/ds
    result=read(base/'results.json')
    assert read(base/'progress.json')['complete']
    ptiming[ds]=read(base/'serial_timing.json')
    arrays=np.load(base/'per_query_metrics.npz');rank_meta=np.load(base/'rankings.npz')
    ours=np.load(HERE/'results'/cfg['label']/'holdout_metrics.npz');mask=ours['outer_holdout']
    assert np.array_equal(ours['query_ids'],rank_meta['query_ids'])
    assert np.array_equal(mask,rank_meta['outer_mask'])
    P[ds]={};paired_pprag[ds]={}
    matched=cfg['semantic_depth']*C['semantic']['probes']
    specs={'caprise_no_dp':(0.,100),'dp002_heuristic':(.02,None),'dp0033_heuristic':(.033,None),'dp0033_matched':(.033,matched)}
    for key,(rad,budget) in specs.items():
        rows=sorted([r for r in result['rows'] if r['beta']==.1 and not r['sensitivity_only'] and r['mean_radius']==rad and r['kprime']==(r['eq21_fixed_budget'] if budget is None else budget)],key=lambda r:r['repeat'])
        assert len(rows)==3,(ds,key,len(rows))
        values=np.stack([arrays[r['name']+f'_k{r["kprime"]}__ndcg10'] for r in rows])
        P[ds][key]={'kprime':rows[0]['kprime'],'mean_radius':rad,'epsilon':rows[0]['epsilon'],
                   'heldout_ndcg10':float(values[:,mask].mean()),
                   'heldout_recall100':float(np.mean([r['metrics_outer']['recall100'] for r in rows])),
                   'candidate_recall':float(np.mean([r['metrics_outer']['candidate_recall'] for r in rows]))}
        paired_pprag[ds][key]=paired(values,ours['duetrank'],mask)
    timing[ds]={}
    for mode in ('ccadpe','direct300','direct1000'):
        candidates=[]
        for path in (HERE/'protocol'/'runs').rglob('results.json'):
            r=read(path)
            if r.get('dataset')==ds and r.get('lexical_mode')==mode and r.get('semantic_enabled'):
                candidates.append((path,r))
        assert len(candidates)==1,(ds,mode,[str(x[0]) for x in candidates])
        path,r=candidates[0]
        assert r['client_pid']!=r['server_pid']
        assert r['config_sha256']==Q['configuration_sha256']
        obs=[o for o in r['observations'] if o['phase']=='measurement']
        assert len(obs)==48 and len({o['query_id'] for o in obs})==16
        assert r['warmup_passes']==2 and r['measurement_passes']==3
        t={'source':str(path),'scope':r['scope'],'mean_ms':r['end_to_end_ms']['mean'],'p50_ms':r['end_to_end_ms']['p50'],'p95_ms':r['end_to_end_ms']['p95'],
           'unique_queries':16,'observations':48,'upload_bytes_mean':float(np.mean([o['actual_application_upload_bytes'] for o in obs])),
           'download_bytes_mean':float(np.mean([o['actual_application_download_bytes'] for o in obs])),
           'encoder_device':r['encoder_device'],'server_backend':r['server_backend'],
           'mean_fetched_documents':float(np.mean([o['fetched_documents'] for o in obs])),
           'per_query_mean_ms':{qid:float(np.mean([o['end_to_end_ms'] for o in obs if o['query_id']==qid])) for qid in sorted({o['query_id'] for o in obs})}}
        timing[ds][mode]=t

summary={'complete':True,'config_sha256':Q['configuration_sha256'],'quality_controls':Q,'pprag':P,'duet_minus_pprag':paired_pprag,
         'loopback_protocol':timing,'pprag_serial_numerical_components':ptiming,
         'protocol_postprocessing':PROTOCOL,
         'scope_notes':['Full quality workloads: 3452 NQ, 7405 HotpotQA, 6980 MS MARCO; three query-noise repeats.',
            'Fusion quality uses fixed SHA256 outer holdouts only; models/hyperparameters selected within repeat-zero calibration.',
            'End-to-end retrieval excludes LLM generation and is warm-process localhost TCP over a sample-scoped encrypted record store.',
            'DuetER/direct loopback protocol: 16 unique latency queries, repeated three times; 48 observations are not 48 independent query samples.',
            'ppRAG is an independent numerical simulation with ideal native-vector access, not official code or matched privacy.',
            'ppRAG serial numerical timing excludes encoding, AES, network and generation; no end-to-end speedup ratio is inferred.',
            'Cached encoder-normalized vectors are scored by FP32 inner product without renormalization; internal encoder dtype/device is recorded separately.']}
(HERE/'experiment_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

def table(caption,label,columns,header,rows,note=''):
    return '\n'.join([r'\begin{table*}[t]',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\centering\small',r'\begin{tabular}{'+columns+'}',r'\toprule',header+r' \\',r'\midrule',*[r' & '.join(row)+r' \\' for row in rows],r'\bottomrule',r'\end{tabular}',r'\parbox{\linewidth}{\vspace{.1cm}\scriptsize '+note+'}' if note else '',r'\end{table*}'])
def mean(ds,mode,key):return Q['datasets'][ds]['methods'][mode][key]['heldout_ndcg10']
quality_rows=[]
for label,mode,key in [('Semantic CCADPE','duet','semantic'),('Lexical CCADPE, cap 300','duet','lexical'),(r'\system + DuetRank','duet','duetrank'),('Direct BM25, cap 300','direct300','lexical'),('Direct BM25, cap 1,000','direct1000','lexical'),('Semantic + direct BM25 300 + DuetRank','direct300','duetrank'),('Semantic + direct BM25 1,000 + DuetRank','direct1000','duetrank')]:
    quality_rows.append([label,*[f'{mean(ds,mode,key):.5f}' for ds in DS]])
for key,label in [('caprise_no_dp','ppRAG-inspired CAPRISE, no DP (ablation)'),('dp0033_matched',r'ppRAG-inspired, DP $.033$, matched semantic slots')]:
    quality_rows.append([label,*[f'{P[ds][key]["heldout_ndcg10"]:.5f}' for ds in DS]])

parts=[r'''\subsection{RQ1: Complete-Corpus Utility and Controlled Comparisons}
\label{sec:rq1-utility}
\paragraph{Matched rerun.}
Table~\ref{tab:unified-quality} uses the same corpora, query holdouts, and encoder inputs for every new result. DuetER and its direct controls reuse fixed document indexes; the ppRAG-inspired control constructs its own full-corpus CAPRISE index. Both DuetER paths use query-noise radius .0375; all datasets use the same lexical top-64-per-partition, planner-order, cap-300 policy. Models are calibrated without the outer labels and frozen across three repeats. All native semantic scores are FP32 inner products over the cached encoder-normalized vectors, without renormalization.
''',table('New complete-corpus quality on fixed outer holdouts; nDCG@10 averaged over three query-noise repeats.','tab:unified-quality','lrrr','Method & NQ & HotpotQA & MS MARCO',quality_rows,
    'Holdouts contain 1,739/3,740/3,526 queries. Direct BM25 uses the identical FTS5 planner and native scorer, with CCADPE filtering omitted. Its fused controls use the same semantic rankings and calibration procedure, with separately fitted models. ppRAG-inspired rows use the same encoder and complete corpora; they do not imply matched privacy or an official reproduction.'),
    r'''\paragraph{What the direct control establishes.}
Hybrid fusion improves over the semantic path on all three datasets. However, the direct-300 fused control has a slightly higher point estimate than \system on every dataset. The negative paired difference on HotpotQA excludes zero, while NQ and MS MARCO intervals include zero. The lexical-only CCADPE filter likewise has lower point estimates than direct-300 on all three datasets, with intervals excluding zero for HotpotQA and MS MARCO. Thus these experiments support the benefit of lexical evidence and learned fusion, but do not establish a quality advantage for lexical CCADPE filtering. Increasing direct BM25 to 1,000 scored candidates is a separate higher-compute control, not a same-work comparison.
''']
def interval(v):return f'{v["mean_difference"]:+.5f} [{v["ci95"][0]:+.5f}, {v["ci95"][1]:+.5f}]'
rows=[]
for ds in DS:
    c=Q['datasets'][ds]['paired_comparisons']
    rows.append([names[ds],interval(c['duet_fusion_minus_semantic']),interval(c['duet_fusion_minus_direct300_fusion'])])
parts.append(table(r'Paired nDCG@10 differences (system minus control), with 95\% bootstrap intervals.','tab:paired-controls','lrr','Data & vs. semantic CCADPE & vs. direct-300 fusion',rows,
    '5,000 paired query resamples after averaging the three repeats per query. Intervals are conditional on the fitted calibration models, not uncertainty over all possible training sets.'))
rows=[]
for ds in DS:
    cfg=C['datasets'][ds];l=read(HERE/'results'/cfg['label']/'lexical.json');s=read(HERE/'results'/cfg['label']/'semantic.json')
    rows.append([names[ds],f'{s["candidates"]["mean"]:.2f}',f'{l["pre_cap_candidates"]["mean"]:.2f}',f'{l["exact_input_candidates"]["mean"]:.2f}',str(cfg['semantic_depth']*128)])
parts.append(table('Executed candidate counts in the new full quality workloads.','tab:unified-counts','lrrrr','Data & Semantic union & Lexical survivors & BM25 inputs & Semantic slots',rows,
    'Lexical responses always contain 1,024 slots before client decryption and the cap of 300. Real survivors and scored documents differ from padded slot counts.'))
parts.append(r'''\paragraph{ppRAG simulation and fidelity limits.}
We follow the public ppRAG preprint's scale-plus-ball-noise CAPRISE and full-corpus search/client-rerank flow, without introducing DuetER's rotation or IVF into that baseline. The controlled setting shares Granite with DuetER, $s=3$, and $\beta=.1$; the source uses gtr-t5-base. DistanceDP uses an isotropic direction and Gamma-distributed radius with mean .02 or .033; for dimension 384, $\epsilon=384/\bar r$. We report a formula-inspired fixed expansion budget plus a maximum semantic-slot budget of 2,048/2,048/4,096, before deduplication. These choices are independent of qrels. The expansion heuristic does not reproduce the source's Table III exactly and provides no empirical-corpus coverage guarantee. The source's $\beta=.2$ is separately checked on 64 predetermined outer queries per dataset.\footnote{Algorithm source: \url{https://arxiv.org/abs/2601.12331v1}; DistanceDP source: \url{https://arxiv.org/abs/2412.12775}.}

No official implementation was located in this audit. The independent numerical implementation uses seeded pseudorandom noise, FP16 server coordinates, FP32 distance accumulation with TF32 disabled, and direct access to the original FP32 native vectors for ideal client reranking; it does not model FP16 ciphertext-decryption error or execute cryptographic PRF reconstruction, AES payload handling, networking, or generation. Consequently it is an algorithmic utility control, not a claimed official reproduction, equal-privacy comparison, or complete ppRAG deployment. The no-DP row is a CAPRISE ablation, not the full ppRAG method.

At mean DP radius .033 and the maximum slot budgets, changing $\beta$ from .1 to .2 leaves paired nDCG@10 and R@100 unchanged on the same 64 outer queries, averaged over three repeats, in all three corpora. HotpotQA candidate recall decreases from .91406 to .90625 (paired difference $-.00781$, 95\% query-bootstrap interval $[-.02344,0]$); NQ and MS MARCO remain .984375. This limited sensitivity does not establish full-workload equivalence. Additional fixed-budget sweeps at 100 and 200 candidates are retained in the experiment artifacts.
''')
rows=[]
for ds in DS:
    for key,label in [('dp002_heuristic',r'$.02$, heuristic'),('dp0033_heuristic',r'$.033$, heuristic'),('dp0033_matched',r'$.033$, matched slots')]:
        v=P[ds][key];rows.append([names[ds],label,str(v['kprime']),f'{v["heldout_ndcg10"]:.5f}',f'{v["heldout_recall100"]:.5f}'])
parts.append(table('ppRAG-inspired utility under predetermined expansion budgets.','tab:pprag-controlled','llrrr',r'Data & Mean DP radius / policy & $k^{\prime}$ & nDCG@10 & R@100',rows,
    r'All rows use the full held-out workload and three repeats at controlled $\beta=.1$. Matched slots means the maximum pre-deduplication semantic return budget, not equal unique candidates per query. The nominal epsilon values are large, and do not match any DuetER privacy guarantee. The separate 64-query sensitivity is not substituted for these full-workload results.'))
parts.append(r'''\subsection{RQ2: Complete Retrieval Timing and Its Scope}
\label{sec:rq2-efficiency}
\paragraph{One real client wall-clock interval.}
The new measurement starts with raw query text and ends after the top-20 fetched documents have been authenticated, decrypted, parsed, and checked. Two separate processes communicate over persistent localhost TCP: Search returns protected records, and Fetch returns encrypted full documents. The clock includes the real query encoder, query transforms, HMAC probes, serialization, server IVF/FTS5 and local refinement, fixed-slot record lookup/transfer, all payload and join-capsule decryptions, native reranking, DuetRank, and Fetch. Overlapping client RPC and server spans are not added together as a synthetic total.

Both encoder and interactive server run on CPU in this measurement; the offline quality runner's GPU batch times are not substituted for latency. Each dataset uses 16 fixed, uniformly spaced queries, two full warmup passes, and three measured passes. Thus there are 48 measured executions but only 16 distinct queries. Query noise is fixed to repeat zero for paired timing, while each request recomputes its embedding and retrieval. Modes run in a fixed order, without independent process-run replications; small latency differences are not evidence of stable superiority. Models, process startup, TCP connection setup, and offline record preparation are excluded. This measures warm process/OS-cache behavior on loopback, not cold-cache or WAN performance; it excludes LLM generation.
''')
rows=[]
for ds in DS:
    for mode,label in [('ccadpe',r'\system'),('direct300','Direct-300 fusion'),('direct1000','Direct-1,000 fusion')]:
        t=timing[ds][mode];rows.append([names[ds],label,f'{t["mean_ms"]:.2f}',f'{t["p50_ms"]:.2f}',f'{t["p95_ms"]:.2f}',f'{t["download_bytes_mean"]/2**20:.3f}'])
parts.append(table('Measured complete retrieval on loopback, with real encrypted payload and document transfer.','tab:unified-protocol','llrrrr','Data & Pipeline & Mean ms & p50 ms & p95 ms & Download MiB',rows,
    'Each row uses 16 queries times three measured passes after two warmup passes. Downloads are actual application-layer bytes including framing, not TCP/IP/link-layer traffic. Direct controls use their own frozen calibrated fusion models and the same semantic branch.'))
parts.append(r'''\paragraph{Protected records and storage scope.}
Every timed request reruns index lookups and local scoring. FTS5 and semantic IVF cover the complete corpus; the reused lexical coordinate cache covers the recorded workload's candidate union, rather than a newly built coordinate index for arbitrary queries. Protected payloads are prepared only for the fixed timing workload and its documented coverage margin; missing coordinates or records cause failure rather than plaintext fallback or replay of saved rankings. Semantic payloads contain the original 384-dimensional FP32 vectors. Lexical payloads contain complete sequences of owner-assigned injective uint32 token IDs padded to 256-byte buckets, so the client reconstructs native term counts. Fresh AES-GCM keys protect payloads, join capsules, and full documents. For semantic records, the server lookup key is a cell-local position and the source-row mapping remains owner/client-side. The lexical implementation retains stable FTS5 and corpus-row lookup identifiers; their exposure is included in the harness's lexical leakage, and removal of the legacy original-ID mapping table alone does not anonymize them. The harness still does not constitute a production isolation or security evaluation.

The online CPU encoder is actually executed for each request. Its vectors can differ numerically from the retained CUDA-encoded quality inputs. Separate diagnostics compare the fresh and cached vectors and the resulting rankings after timing; cached vectors are never substituted into a measured request. Cached-input validation passes all 144 query/mode functional comparisons, including native rankings and final top-20 outputs. It still records 3/13/23 differing query-cell candidate sets among 2,048 comparisons per dataset. Replaying all 39 original GPU batches reproduces their selections with identical FP16 query bytes, confirming FP32 backend cutoff differences; it does not establish bitwise equality of all intermediate candidates. Consequently full-workload quality and sampled online timing share explicit algorithmic parameters, while their backend/numerical limits remain visible.

In the separate fresh-encoder diagnostic, all nine modes have the same mean nDCG@10 as their cached-input references to numerical precision on the 16-query timing sample. For \system, these sample means are .52605/.67962/.37349; the mean top-20 set intersections are 19.9375/19.5000/19.8125, while only 6/6/10 of the 16 queries retain the identical order. Each fresh output is stable across the three timed passes. The samples contain only 7/5/10 outer-holdout queries and otherwise include calibration queries, so these observations diagnose backend drift and do not replace the full held-out results.

\paragraph{Observed implementation cost.}
The mean protected-record SQLite lookup spans for \system are 537.91/537.87/928.96 ms on NQ/HotpotQA/MS MARCO. They account for approximately 70\%/57\%/60\% of the respective client wall-clock totals in this implementation. These are actual record-by-record accesses; no mid-run cache or batching optimization was applied. This storage cost limits conclusions about the retrieval algorithm's intrinsic speed. Server and RPC spans overlap and must not be added together.

\paragraph{Separate ppRAG numerical components.}
The ppRAG-inspired control additionally measures serial full-corpus search and native reranking on 16 predetermined outer queries in one timed pass, after two warmup requests using the first two selected queries. That timer includes index reads, CPU-to-GPU transfer, distance scoring, top-$k^{\prime}$ selection, and native-vector memory-map page reads and reranking. The native corpus is not fully preloaded, so the client span includes storage access rather than pure arithmetic. It excludes query encoding/preparation, payload cryptography, network, and generation. Table~\ref{tab:pprag-serial} is therefore not compared as an end-to-end speedup against Table~\ref{tab:unified-protocol}.
''')
rows=[]
for ds in DS:
    v=ptiming[ds]['summary']['total_seconds'];rows.append([names[ds],str(ptiming[ds]['kprime']),f'{v["mean"]*1000:.2f}',f'{v["p50"]*1000:.2f}',f'{v["p95"]*1000:.2f}'])
parts.append(table('Separately scoped ppRAG-inspired serial numerical search plus native rerank; not full-protocol latency.','tab:pprag-serial','lrrrr',r'Data & $k^{\prime}$ & Mean ms & p50 ms & p95 ms',rows,
    'RTX 4060 full-corpus scans; the complete ciphertext corpus is not kept resident on the GPU. Query preparation, AES, network, and generation are excluded. These 16 outer queries differ from the loopback sample.'))
(HERE/'manuscript'/'unified_results.tex').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')

lines=['实验结果与修改说明（2026-11-07）','', '全部结果保存于本独立目录；旧论文与历史实验保留。',
       '一、统一口径', '查询噪声半径0.0375；数据库噪声0.1125。词法16分区×64返回槽；FTS5最多1000候选，恢复原planner顺序后最多300进入精确BM25。预算保留强制首词的例外。',
       '语义为缓存encoder-normalized向量的FP32点积，不额外重新归一化；不再将有限精度实现称作严格精确余弦。',
       '全量3452/7405/6980查询，三次噪声重复；质量报告只使用相同的1739/3740/3526 outer holdout。仅校准集选择融合参数。','',
       '二、主要质量结果（nDCG@10，三次均值）','数据集\t语义单路\tDuetER\t直接BM25-300+融合\t直接BM25-1000+融合\tppRAG仿真DP.033同语义槽预算']
for ds in DS:lines.append('\t'.join([names[ds],*[f'{x:.8f}' for x in [mean(ds,'duet','semantic'),mean(ds,'duet','duetrank'),mean(ds,'direct300','duetrank'),mean(ds,'direct1000','duetrank'),P[ds]['dp0033_matched']['heldout_ndcg10']]]]))
lines+=['','关键解释：混合检索相对语义单路的收益仍然存在，但直接BM25-300加同样融合流程的均值在三库均略高；HotpotQA的差异95%置信区间不跨0。不能将混合检索收益归因于CCADPE候选筛选。',
        '直接BM25-1000是额外计算量对照。ppRAG为独立仿真，保留无DP消融、DP半径/预算对照，以及64条outer query的原文beta=.2敏感性；并非官方代码或等隐私比较。','',
        '三、完整检索计时','实际执行CPU编码、查询准备、两个检索分支、两进程TCP传输、AES-GCM解密、精排、融合与Top20 Fetch。每库16个不同query，2轮warmup、3轮measurement。该测试是warm loopback，不包含LLM生成/WAN。',
        '数据集\t流程\tmean/p50/p95(ms)\t平均下载(MiB)']
for ds in DS:
    for mode in ('ccadpe','direct300','direct1000'):
        t=timing[ds][mode];lines.append(f'{names[ds]}\t{mode}\t{t["mean_ms"]:.2f}/{t["p50_ms"]:.2f}/{t["p95_ms"]:.2f}\t{t["download_bytes_mean"]/2**20:.3f}')
lines+=['','计时保护记录只为所选query及覆盖范围预建，但在线实际重新检索，无排名回放；FTS5副本去除了原始文档ID旁表，semantic server仅使用cell-local位置。仅凭此harness不声称生产安全部署完成。',
        '九组新CPU编码与缓存输入在各16条计时query上的平均nDCG@10一致到浮点精度；但Top20顺序并不完全一致。这些样本含calibration query，不替代全量holdout质量。词法服务端仍保留稳定FTS5/corpus-row标识，不能将去除原ID旁表说成完全匿名化。',
        '逐条读取受保护SQLite记录是当前主要开销：DuetER三库均值538/538/929 ms，约占总计时70%/57%/60%。各模式固定顺序运行一次，未重复独立进程轮次；不把小幅差异解释为稳定优势。',
        'ppRAG另测了串行GPU数值检索+精排组件；它排除AES/编码/网络，因此没有用其与DuetER完整流程计算加速比。','',
        '四、论文修改','新质量、直接BM25和ppRAG对照成为正文实验；旧质量/计时移入历史附录并明确配置边界。候选伪代码和参数表与代码统一；量化posting分析仅保留为未执行的可选理论分支。保留此前收缩后的安全结论，未把旧攻击数据当作新噪声配置的安全验证。',
        '原始OpenReview最终PDF绑定和上一版修订保留。新的PDF与LaTeX源码另存。','',
        '证据：config.json，quality_comparison.json，experiment_summary.json，pprag/summary.json，protocol/runs/*/results.json；运行与一致性核验见对应manifest和validation文件。']
(HERE/'实验结果与修改说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Final numerical report and manuscript input generated.')
