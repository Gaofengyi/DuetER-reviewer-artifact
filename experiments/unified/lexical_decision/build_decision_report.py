"""A decision from frozen evidence; no new retrieval, fitting, or timing."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
config = read(ROOT/'config.json')
quality = read(ROOT/'quality_comparison.json')
protocol = read(ROOT/'protocol'/'PROTOCOL_TIMING_SUMMARY.json')
assert protocol['status']=='complete'
assert quality['configuration_sha256']==sha(ROOT/'config.json')==protocol['config_sha256']
names={'nq':'NQ','hotpotqa':'HotpotQA','msmarco':'MS MARCO'}
sources=[ROOT/'config.json',ROOT/'quality_comparison.json',ROOT/'protocol'/'PROTOCOL_TIMING_SUMMARY.json',
         ROOT/'protocol'/'server.py',ROOT/'protocol'/'client.py',ROOT/'protocol'/'records.py']
data={}
for ds,cfg in config['datasets'].items():
    q=quality['datasets'][ds]
    local_path=ROOT/'results'/cfg['label']/'lexical.json'
    native_path=ROOT/'direct_bm25'/ds/'quality.json'
    local,native=read(local_path),read(native_path)
    sources.extend([local_path,native_path])
    planner_mean=native['controls']['1000']['scored_candidates_mean']
    survivors=local['pre_cap_candidates']['mean']
    t={r['lexical_mode']:r for r in protocol['runs'] if r['dataset']==ds}
    current,direct=t['ccadpe'],t['direct300']
    cc_time=current['end_to_end_ms']['mean']; d_time=direct['end_to_end_ms']['mean']
    cc_down=current['application_bytes']['actual_application_download_bytes']['mean']
    d_down=direct['application_bytes']['actual_application_download_bytes']['mean']
    data[ds]={
        'name':names[ds], 'quality_holdout_queries':q['holdout_count'],
        'quality_means':{mode:{branch:q['methods'][mode][branch]['heldout_ndcg10']
                              for branch in ('lexical','duetrank')}
                         for mode in ('duet','direct300','direct1000')},
        'ccadpe_minus_direct300':{branch:q['paired_comparisons'][key] for branch,key in
            [('lexical','duet_lexical_minus_direct300'),('fused','duet_fusion_minus_direct300_fusion')]},
        'ccadpe_query_metric_outcomes':{branch:{name:round(q['paired_comparisons'][key][name+'_fraction']*q['holdout_count'])
            for name in ('win','tie','loss')} for branch,key in
            [('lexical','duet_lexical_minus_direct300'),('fused','duet_fusion_minus_direct300_fusion')]},
        'candidates':{'planner_mean':planner_mean,'ccadpe_survivors_mean':survivors,
            'ccadpe_removal_fraction_of_all_planner_occurrences':1-survivors/planner_mean,
            'ccadpe_exact_input_mean':local['exact_input_candidates']['mean'],
            'direct300_exact_input_mean':native['controls']['300']['scored_candidates_mean'],
            'ccadpe_transmitted_lexical_slots':1024,'direct300_transmitted_lexical_slots':300,
            'fixed_slot_reduction_fraction':1-300/1024},
        'measured_loopback':{'ccadpe_mean_ms':cc_time,'direct300_mean_ms':d_time,
            'direct300_minus_ccadpe_ms':d_time-cc_time,
            'direct300_time_reduction_fraction':1-d_time/cc_time,
            'ccadpe_download_mib':cc_down/2**20,'direct300_download_mib':d_down/2**20,
            'direct300_download_reduction_fraction':1-d_down/cc_down,
            'ccadpe_record_lookup_ms':current['stage_ms']['search_server.record_lookup_ms']['mean'],
            'direct300_record_lookup_ms':direct['stage_ms']['search_server.record_lookup_ms']['mean'],
            'ccadpe_semantic_scan_ms':current['stage_ms']['search_server.semantic_lookup_refine_ms']['mean'],
            'direct300_semantic_scan_ms':direct['stage_ms']['search_server.semantic_lookup_refine_ms']['mean']}}

report={
    'status':'decision_complete', 'decision_date':'2026-11-07',
    'decision':'Remove the current lexical CCADPE refinement from the default research architecture; retain it as a historical/ablation variant.',
    'replacement':'Same budgeted HMAC-FTS5 planner (<=1000) -> planner first <=300 -> 300 padded protected records -> authenticated client decryption -> exact native BM25 -> existing semantic branch and separately calibrated fusion -> protected Top20 Fetch.',
    'scope':'Lexical CCADPE only. Does not decide the value or safety of semantic CCADPE.',
    'config_sha256':sha(ROOT/'config.json'),'datasets':data,
    'decision_basis':['No demonstrated aggregate quality benefit over direct300, including lexical-only comparison without fusion-model confounding.',
        'Lower direct300 point-estimate latency on two corpora; small reverse difference on MS MARCO is not a stable mode advantage under fixed-order single-run measurement.',
        'Direct300 reduces measured total download on all three corpora and reduces fixed lexical record slots from 1024 to 300.',
        'Native feature and document protection are provided by existing AEAD wrappers retained in the direct route, not by the lexical distance-refinement step.',
        'Removing lexical numerical coordinates removes an observable geometric channel in a clean implementation; no whole-transcript privacy dominance is proven.',
        'CCADPE removes few planner candidates, after which the separate planner-order cap performs most of the reduction.'],
    'limits':['Quality intervals are conditional on fixed calibration models; three repeats average query noise, not independent document-index seeds.',
        'NQ and MS fused intervals include zero; do not call direct300 strictly statistically better everywhere or infer equivalence from non-significance.',
        'Some queries improve with CCADPE; aggregate removal is an architecture decision, not universal per-query dominance.',
        'Latency uses 16 unique queries, 3 measured passes, 2 warmup passes, CPU and loopback. Modes ran in fixed order once; storage/cache effects remain.',
        'Existing direct controls still load legacy lexical coordinate/partition caches. Bypassing scoring is measured; a clean coordinate-free server has not been benchmarked.',
        'HMAC-FTS5 continues to expose declared equality, incidence, frequencies, lengths, positions and access patterns; eliminating CCADPE does not eliminate those leakages.',
        'Changed candidates and Fetch outputs prevent asserting whole-transcript privacy dominance without a dedicated argument.',
        'The current heldout results informed this architecture decision. Future performance confirmation should use a new independent evaluation or disclose post-selection reuse.',
        'The controls are matched budgeted-planner BM25, not unbudgeted full-term global BM25.'],
    'engineering_next':['Introduce an explicit direct300 main profile without overwriting frozen benchmark config or results.',
        'Separate direct record lookup/alias handling from legacy lexical partition assignments; retain protected payload, join, fixed-slot padding and Fetch.',
        'Do not build/store/load lexical CCADPE document coordinates or keys in the simplified server; omit lexical query views and transforms.',
        'Revalidate selection order, native BM25, cross-path join and final outputs; then measure the cleaned implementation on independent queries.',
        'Revise method/algorithms/contributions to distinguish semantic distance ranking from the keyed lexical planner and exact client scoring; keep negative ablation results.'],
    'reintroduction_gate':'Only reconsider a newly specified variant if a preregistered matched comparison shows a meaningful quality/cost or precisely defined privacy benefit with acceptable tradeoffs; validate on independent data. Existing evidence does not meet this gate.',
    'source_files':[{'path':str(p),'sha256':sha(p)} for p in sources]}
(HERE/'decision.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

lines=['词法 CCADPE 保留决策（2026-11-07）','',
    '决定：当前词法 CCADPE 分区筛选不作为主方案的必要组件；采用直接 BM25-300 路线作为默认方向。旧实现和全部结果作为消融与研究记录保留。',
    '范围：只针对词法分支，不据此否定或认证语义 CCADPE。本次是既有证据的决策分析，没有新增检索、训练或计时。','',
    '一、同候选精排预算的质量',
    '数据集\t词法CCADPE\t直接BM25-300\tCCADPE双路融合\t直接300双路融合\t融合差值(CCADPE-直接300)及95%CI']
for ds,v in data.items():
    m=v['quality_means'];d=v['ccadpe_minus_direct300']['fused']
    lines.append(f"{v['name']}\t{m['duet']['lexical']:.6f}\t{m['direct300']['lexical']:.6f}\t{m['duet']['duetrank']:.6f}\t{m['direct300']['duetrank']:.6f}\t{d['mean_difference']:+.6f} [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]")
lines+=['融合质量三库均值均略偏向直接300，HotpotQA配对差异区间不跨0；NQ/MS区间跨0，不能声称处处显著更好或严格等价。',
    '词法单路同样没有CCADPE的均值优势，HotpotQA/MS区间不跨0，因而该结论不完全依赖重新校准融合模型。',
    '各查询有赢有输；移出默认方案并不等于每条query直接300都更好。平局仅指nDCG相同，不代表排名相同。',
    '数据集\t词法CCADPE胜/平/负\t融合CCADPE胜/平/负']
for v in data.values():
    counts=v['ccadpe_query_metric_outcomes']
    lines.append(v['name']+'\t'+'\t'.join('/'.join(str(counts[b][x]) for x in ('win','tie','loss')) for b in ('lexical','fused')))
lines+=['',
    '二、真实流程代价（现有warm-loopback测量）',
    '数据集\tCCADPE/直接300均值ms\t直接300耗时降低%\tCCADPE/直接300下载MiB\t下载降低%']
for v in data.values():
    t=v['measured_loopback'];lines.append(f"{v['name']}\t{t['ccadpe_mean_ms']:.2f}/{t['direct300_mean_ms']:.2f}\t{t['direct300_time_reduction_fraction']*100:+.2f}\t{t['ccadpe_download_mib']:.4f}/{t['direct300_download_mib']:.4f}\t{t['direct300_download_reduction_fraction']*100:.2f}")
lines+=['每库只有16个不同查询、两轮预热和三轮测量；固定模式顺序且无独立进程重跑。MS直接300约3%的反向差异不能认定为稳定劣势。',
    '三库总下载均降低。固定词法槽位从1024降至300（减少70.70%），不等于整个系统下载减少70.70%；语义响应和Fetch仍占大部分字节。','',
    '三、筛选机制为何缺少保留依据',
    '数据集\tplanner真实候选均值\tCCADPE幸存均值\tCCADPE实际剔除占比%']
for v in data.values():
    c=v['candidates'];lines.append(f"{v['name']}\t{c['planner_mean']:.2f}\t{c['ccadpe_survivors_mean']:.2f}\t{c['ccadpe_removal_fraction_of_all_planner_occurrences']*100:.2f}")
lines+=['当前每分区64个返回位，总计1024槽，已经接近planner最多1000条；局部筛选只排除少量候选。随后仍按原planner顺序截至300，主要数量压缩来自这一步。',
    '云端跨分区距离没有统一用于选全局前300。因此本结构增加坐标变换、局部筛选、受保护记录取得与传输，而没有展示独立质量收益。','',
    '四、移除筛选不等于明文检索',
    '保留HMAC词项标签/预算FTS5索引、受保护的原生词法载荷、别名和加密join、固定300槽、客户端认证解密与精确BM25、密文正文Fetch。',
    '移除词法文档/查询数值坐标及局部距离计算，可消除这一几何观测通道；但HMAC索引仍泄漏相等性、posting关联与统计、访问轨迹等。',
    '候选与Fetch输出会改变，不能由此声称完整服务器视图下已严格证明更安全。现有direct实现仍加载旧坐标/分区cache；后续需要真正解除依赖，现阶段不把这些存储项算作已实现节省。','',
    '五、后续落地和重新保留门槛',
    '默认路线：原预算planner最多1000 → 原顺序前300 → 300个补齐槽的加密记录 → 客户端精确BM25 → 与语义路融合 → Top20 Fetch。',
    '1000候选作为较大计算量控制，旧CCADPE作为消融保留。实现简化需要独立配置及来源记录，避免改写本轮冻结结果。',
    '只有新变体在同等索引/载荷/评分与明确预算下展示有意义的质量、成本或具体隐私收益，并通过独立数据验证，才考虑恢复为主组件。',
    '当前评估集已经用于架构取舍，下一轮确认应使用新的独立评估，或透明披露复用了用于选择方案的数据。',
    '论文代价：需重写“统一双路距离保持接口”的主贡献，不能将拿掉模块本身包装为创新；后续独立贡献仍需机制控制和安全证据。','',
    '数据与逐项来源指纹：同目录decision.json。原质量、计时、代码和论文保持其既有版本。']
(HERE/'词法CCADPE决策说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'decision':report['decision'],'rows':{k:{'quality':v['quality_means'],
    'candidates':v['candidates'],'timing':v['measured_loopback']} for k,v in data.items()}},ensure_ascii=False,indent=2))
