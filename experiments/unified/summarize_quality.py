"""Summarize completed unified quality artifacts; no retrieval or training."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
cfg = read(ROOT / "config.json")
verification = read(ROOT / "quality_final_verification.json")
assert verification["status"] == "passed"
config_hash = hashlib.sha256((ROOT / "config.json").read_bytes()).hexdigest()
assert config_hash == verification["config_sha256"]
summary = {
    "status": "complete_and_verified",
    "config_sha256": config_hash,
    "scope": "Complete-query retrieval utility, not protocol latency; fixed document/query embeddings and deterministic planner candidates reused, all query noise/local refinement/native scoring/fusion recomputed.",
    "model_policy": "One repeat-zero calibration-selected model per dataset; frozen for three query-noise repeats. Quality scored only on outer holdout; current-repeat features only.",
    "bootstrap_scope": "5000 paired-query resamples after averaging per-query differences across repeats; conditional on one trained model, not training-seed uncertainty.",
    "verification_path": str(ROOT / "quality_final_verification.json"),
    "datasets": {},
}
lines = [
    "统一配置全量质量重跑：完成与核验报告（2026-11-07）",
    "",
    "1. 范围与配置",
    f"配置 SHA256：{config_hash}",
    "两路 query radius=.0375，database radius=.1125；语义 work/local=512/256，128 probes，深度16/16/32；词法 work/local=8192/32，16 partitions，local64，planner<=1000，按原 planner 顺序取存活者前300，再计算 native exact BM25。",
    "使用固定文档索引、查询 embedding、原始确定性 planner 候选作为输入；新执行查询噪声、局部检索、精排和融合。没有以历史局部排名或融合排名充当本轮结果。该质量实验没有重跑全部在线 encoder/FTS 查询，也不是端到端计时。",
    "三次重复仅更换 query-noise seed；repeat0 calibration 内选择超参数和回退，再在全部 repeat0 calibration 上拟合一次，模型冻结评估三个 outer holdout repeats。每次特征只使用本次检索，不要求跨重复聚合。",
    "",
    "2. 同一 outer holdout 上的 nDCG@10（三重复均值）",
]
for name, dc in cfg["datasets"].items():
    dest = ROOT / "results" / dc["label"]
    f = read(dest / "fusion.json")
    s = read(dest / "semantic.json")
    l = read(dest / "lexical.json")
    metrics = {key: value["mean_ndcg_at_10"] for key, value in f["held_out"].items()}
    gain = dict(f["duetrank_vs_semantic"])
    gain["relative_percent"] = 100 * gain["mean_absolute_gain"] / metrics["semantic"]
    summary["datasets"][name] = {
        "query_count": f["query_count"],
        "calibration_queries": f["calibration_queries"],
        "outer_holdout_queries": f["outer_holdout_queries"],
        "repeats": f["repeats"],
        "ndcg_at_10": metrics,
        "duetrank_vs_semantic": gain,
        "duetrank_vs_fixed_rrf": f["duetrank_vs_fixed_rrf"],
        "per_repeat": f["held_out"],
        "selection": f["selection"],
        "semantic_native_scoring_candidates": s["candidates"],
        "lexical_pre_cap_candidates": l["pre_cap_candidates"],
        "lexical_native_scoring_candidates": l["exact_input_candidates"],
        "fusion_source": str(dest / "fusion.json"),
        "verification": verification["datasets"][name],
    }
    lines += [
        f"{name}: 全查询 {f['query_count']}；calibration {f['calibration_queries']}；outer holdout {f['outer_holdout_queries']}；3 repeats。",
        "  semantic={semantic:.9f}, lexical={lexical:.9f}, DuetRank={duetrank:.9f}, calibrated fixed RRF={fixed_rrf:.9f}, standard RRF={standard_rrf:.9f}".format(**metrics),
        f"  DuetRank vs semantic: 绝对增益 {gain['mean_absolute_gain']:.9f}，相对 {gain['relative_percent']:.4f}%；95% CI [{gain['ci95_low']:.9f}, {gain['ci95_high']:.9f}]；单侧 bootstrap p={gain['one_sided_bootstrap_p']:.9f}。",
        f"  实际候选均数：semantic union {s['candidates']['mean']:.4f}；lexical cap前 {l['pre_cap_candidates']['mean']:.4f}；exact BM25输入 {l['exact_input_candidates']['mean']:.4f}。",
    ]
lines += [
    "",
    "置信区间由5000次 paired-query bootstrap 得到：先对每个查询取三次重复的差值均值，再重采样查询。区间条件于一个已训练模型，不衡量重新训练/划分的不确定性。以上全部比较均在同一 outer holdout 上，未与全查询单路指标混减。",
    "",
    "3. 最终核验",
    "三库 × 三重复的 QIDs、outer masks、排名形状、ID范围、去重、保存指标与JSON一致性、候选规则和配置hash全部通过。词法全部53,511 query-repeats验证保留原planner顺序，精确BM25前后候选集合不变。",
    "HotpotQA与MS MARCO各8个均匀查询 ×3 repeats，独立FP64 native cosine复核本轮保存的emitted union，Top100集合及顺序均24/24完全一致；最大阈值偏差0。NQ已由前置真实查询小测完成9次FP64复核，本轮未重复。",
    "历史导入源码的完整SHA256与运行前一致。缓存中小文件完整SHA256、其余大文件精确大小和4096字节头指纹与运行前一致；这不是所有大文件全内容或原工作区所有文件的哈希审计。运行输出仅写新目录，历史索引以只读方式加载。",
    "",
    "4. 离线计算读取调整及计时边界",
    "原顺序处理进程完成NQ全部步骤后，完成HotpotQA cloud，因随机memmap精排受存储读取限制而中止HotpotQA尚未完成的精排。NQ完整结果保留；HotpotQA保留本轮新生成的emitted候选；MS随后执行同一cloud算法。没有重用历史排名。详见 semantic_stage_separation.json。",
    "HotpotQA cloud仅保留stdout四舍五入值38.1秒，未将其伪装为高精度时钟记录。HotpotQA/MS精排改为先顺序装入本库原始embedding，再在本轮候选上计算native cosine；每次只装入一个语料。",
    "这只是离线质量计算的I/O安排，不改变候选/打分/排序规则；独立FP64复核通过。阶段时钟包含不同cache状态与I/O边界，不能求和冒充交互式或协议端到端时延；完整流程计时由单独protocol实验负责。",
    "",
    "5. 使用限制",
    "新旧质量差异不能全部归因于query radius修正：本轮同时固定了FP32评分、基于query ID的随机种子和单次查询融合特征。ppRAG与直接BM25对照由独立结果文件提供，本报告不替其宣称效果。没有运行攻击实验。",
    "",
    "输出：quality_summary.json（可机器读取摘要）；quality_final_verification.json（逐文件及逐库核验）；results/{nq,hotpotqa,msmarco_dev}/fusion.json（指标和CI原始来源）；各目录的duetrank.joblib与排名NPZ。",
]
summary["write_scope"] = verification["historical_write_scope"]
summary["cache_hash_limit"] = verification["cache_hash_limit"]
(ROOT / "quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
(ROOT / "quality_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"status": summary["status"], "outputs": [str(ROOT / "quality_summary.json"), str(ROOT / "quality_report.txt")], "metrics": {k: v["ndcg_at_10"] for k, v in summary["datasets"].items()}}, ensure_ascii=False))
