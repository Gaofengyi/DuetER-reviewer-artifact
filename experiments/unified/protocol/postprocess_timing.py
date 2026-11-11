"""Read-only postprocessing of the completed nine-run timing experiment.

Never invokes the encoder, retrieval server, or a benchmark. Private document
IDs are used only to score already-saved rankings; no keys or document text
are exported. All 48 measurements are retained, with no outlier selection.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from runtime import initialize
CONFIG = initialize()
import numpy as np

ROOT = Path(__file__).resolve().parent
STAGES = (
    "encoder_ms", "request_build_ms", "search_rpc.rpc_wall_ms",
    "search_rpc.request_serialize_ms", "search_rpc.transfer_and_server_ms",
    "search_rpc.response_parse_ms", "search_server.request_parse_ms",
    "search_server.request_decode_vectors_ms", "search_server.semantic_lookup_refine_ms",
    "search_server.lexical_fts5_planner_ms", "search_server.lexical_partition_refine_ms",
    "search_server.direct_candidate_selection_ms", "search_server.record_lookup_ms",
    "payload_join_decrypt_dedup_ms", "semantic_exact_rerank_ms",
    "planner_restore_cap_exact_bm25_ms", "duetrank_ms", "fetch_rpc.rpc_wall_ms",
    "fetch_server.document_lookup_ms", "fetch_decrypt_parse_verify_ms",
)


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "min": float(values.min()),
            "max": float(values.max())}


def nested(item, key):
    for part in key.split("."):
        item = item.get(part, 0.0) if isinstance(item, dict) else 0.0
    return float(item)


def ndcg(rows, docids, truth):
    observed = [truth.get(docids[str(int(row))], 0) for row in rows[:10]]
    ideal = sorted(truth.values(), reverse=True)[:10]
    dcg = sum((2.0 ** gain - 1.0) / math.log2(rank + 2) for rank, gain in enumerate(observed))
    idcg = sum((2.0 ** gain - 1.0) / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def sample_quality(result, measured, private, qrels, qrels_path, private_path):
    reference_path = Path(result["fresh_encoder_vs_cached_quality_postrun"]["fused_reference"])
    grouped = defaultdict(list)
    for item in measured:
        grouped[item["query_id"]].append(item)
    per_query = []
    with np.load(reference_path) as reference:
        rankings, ids = reference["dual"][0], reference["query_ids"]
        holdout = reference["outer_holdout"]
        for query in private["queries"]:
            qid, index = query["id"], query["index"]
            assert str(ids[index]) == qid
            obs = grouped[qid]
            assert len(obs) == 3
            cached = [int(row) for row in rankings[index, :20] if row >= 0]
            fresh = [item["top20"] for item in obs]
            cached_ndcg = ndcg(cached, private["document_ids"], qrels[qid])
            fresh_ndcgs = [ndcg(rows, private["document_ids"], qrels[qid]) for rows in fresh]
            overlap = [len(set(rows) & set(cached)) for rows in fresh]
            exact = [rows == cached for rows in fresh]
            cached_gains = [qrels[qid].get(private["document_ids"][str(row)], 0) for row in cached[:10]]
            gains_equal = all([qrels[qid].get(private["document_ids"][str(row)], 0) for row in rows[:10]] == cached_gains for rows in fresh)
            # Independently confirm the diagnostic that client wrote after timing.
            for item, same, intersection in zip(obs, exact, overlap):
                prior = item["fresh_encoder_vs_cached_quality_postrun"]
                assert prior["top20_exact_match"] == same
                assert prior["top20_set_overlap"] == intersection
            per_query.append({"query_id": qid, "query_index": index,
                "outer_holdout": bool(holdout[index]), "measurement_repetitions": len(obs),
                "cached_ndcg_at_10": cached_ndcg,
                "fresh_ndcg_at_10_mean": float(np.mean(fresh_ndcgs)),
                "fresh_ndcg_at_10_each_pass": fresh_ndcgs,
                "paired_delta_ndcg_at_10": float(np.mean(fresh_ndcgs)) - cached_ndcg,
                "top20_set_intersection_mean": float(np.mean(overlap)),
                "top20_exact_match_count": sum(exact),
                "top20_exact_match_rate": float(np.mean(exact)),
                "top10_relevance_gain_sequence_exact_on_all_passes": gains_equal,
                "fresh_distinct_top20_lists_across_passes": len({tuple(rows) for rows in fresh})})
    aggregate = {"distinct_sample_queries": len(per_query),
        "outer_holdout_sample_queries": sum(item["outer_holdout"] for item in per_query),
        "fresh_ndcg_at_10": float(np.mean([item["fresh_ndcg_at_10_mean"] for item in per_query])),
        "cached_ndcg_at_10": float(np.mean([item["cached_ndcg_at_10"] for item in per_query])),
        "paired_delta_ndcg_at_10": float(np.mean([item["paired_delta_ndcg_at_10"] for item in per_query])),
        "top20_set_intersection_mean": float(np.mean([item["top20_set_intersection_mean"] for item in per_query])),
        "top20_exact_match_rate": float(np.mean([item["top20_exact_match_rate"] for item in per_query])),
        "queries_top20_exact_on_all_three_passes": sum(item["top20_exact_match_count"] == 3 for item in per_query),
        "queries_top10_relevance_gain_sequence_exact_on_all_passes": sum(item["top10_relevance_gain_sequence_exact_on_all_passes"] for item in per_query),
        "max_abs_per_query_ndcg_delta": max(abs(item["paired_delta_ndcg_at_10"]) for item in per_query),
        "all_fresh_top20_stable_across_passes": all(item["fresh_distinct_top20_lists_across_passes"] == 1 for item in per_query)}
    return {"dataset": result["dataset"], "lexical_mode": result["lexical_mode"],
        "scope": "16 evenly spaced queries over all query IDs, including calibration and holdout; small paired diagnostic, not full-query/heldout utility evidence",
        "aggregation": "nDCG uses (2^gain-1)/log2(rank+2), cutoff 10, exactly as original evaluate(); average the three fresh observations within each query, then average 16 queries; cached reference uses noise repeat 0 once per query",
        "sources": {"fused_reference": fingerprint(reference_path), "qrels": fingerprint(qrels_path),
                    "private_docid_mapping": fingerprint(private_path)},
        "aggregate": aggregate, "per_query": per_query}


def main():
    expected = [(dataset, mode) for dataset in ("nq", "hotpotqa", "msmarco")
                for mode in ("ccadpe", "direct300", "direct1000")]
    paths = list((ROOT / "runs").glob("*/results.json"))
    results = {}
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (result["dataset"], result["lexical_mode"])
        assert key not in results, f"duplicate formal run: {key}"
        assert result["semantic_enabled"] is True
        results[key] = (path, result)
    assert set(results) == set(expected)
    source_sets = [result["protocol_source_sha256"] for _, result in results.values()]
    assert all(item == source_sets[0] for item in source_sets), "execution source changed between modes"
    config_sha = fingerprint(ROOT.parent / "config.json")["sha256"]
    summary_runs, diagnostics = [], []
    for dataset in ("nq", "hotpotqa", "msmarco"):
        private_path = ROOT / "prepared" / f"{dataset}_cpu_v1" / "client_private.json"
        private = json.loads(private_path.read_text(encoding="utf-8"))
        qrels_path = Path(CONFIG["original_root"]) / "data" / dataset / "qrels" / (CONFIG["datasets"][dataset]["split"] + ".tsv")
        qrels = defaultdict(dict)
        with qrels_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
        expected_ids = {query["id"] for query in private["queries"]}
        for mode in ("ccadpe", "direct300", "direct1000"):
            path, result = results[(dataset, mode)]
            assert result["config_sha256"] == config_sha
            observations = result["observations"]
            warmup = [item for item in observations if item["phase"] == "warmup"]
            measured = [item for item in observations if item["phase"] == "measurement"]
            assert len(warmup) == 32 and len(measured) == 48
            assert set(item["query_id"] for item in measured) == expected_ids and len(expected_ids) == 16
            assert set(Counter(item["query_id"] for item in measured).values()) == {3}
            assert set(Counter(item["query_id"] for item in warmup).values()) == {2}
            assert len({item["request_id"] for item in observations}) == 80
            raw_path = path.with_suffix(".raw.json")
            raw = json.loads(raw_path.read_text(encoding="utf-8"))["observations"]
            assert [(x["request_id"], x["end_to_end_ms"], x["top20"]) for x in raw] == [(x["request_id"], x["end_to_end_ms"], x["top20"]) for x in observations]
            end_to_end = statistics([item["end_to_end_ms"] for item in measured])
            assert all(abs(end_to_end[key] - result["end_to_end_ms"][key]) < 1e-8 for key in ("mean", "p50", "p95"))
            stages = {key: statistics([nested(item, key) for item in measured]) for key in STAGES}
            log_path = path.parent / "server_requests.jsonl"
            logs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
            for kind in ("search", "fetch"):
                subset = [item for item in logs if item["kind"] == kind and item["request_id"].startswith("measurement-")]
                assert len(subset) == 48
                for field in ("serialize_ms", "send_ms", "server_wall_after_receive_ms"):
                    stages[f"server_log.{kind}.{field}"] = statistics([item[field] for item in subset])
            diagnostic = sample_quality(result, measured, private, qrels, qrels_path, private_path)
            diagnostics.append(diagnostic)
            encoder = [item["encoder_vs_cached_diagnostic_outside_timer"] for item in measured]
            counts = {key: statistics([item[key] for item in measured]) for key in
                      ("semantic_unique_candidates", "lexical_survivors", "lexical_exact_scored", "dummy_slots", "fetched_documents", "fetched_plaintext_bytes")}
            byte_stats = {key: statistics([item[key] for item in measured]) for key in
                          ("actual_application_upload_bytes", "actual_application_download_bytes")}
            summary_runs.append({"dataset": dataset, "lexical_mode": mode, "semantic_enabled": True,
                "source": fingerprint(path), "raw_source": fingerprint(raw_path), "server_log_source": fingerprint(log_path),
                "warmup_observations": 32, "measurement_observations": 48, "distinct_sample_queries": 16,
                "end_to_end_ms": end_to_end, "application_bytes": byte_stats, "stage_ms": stages,
                "record_lookup_fraction_of_total_wall_mean": stages["search_server.record_lookup_ms"]["mean"] / end_to_end["mean"],
                "online_counts": counts, "encoder_parameter_dtype": result["encoder_parameter_dtype"],
                "encoder_output_dtype_before_float32_cast": result["encoder_output_dtype_before_float32_cast"],
                "embedding_diagnostic": {"max_absolute_difference_max": max(item["max_abs"] for item in encoder),
                    "cosine_min": min(item["cosine"] for item in encoder),
                    "fresh_norm_min": min(item["fresh_embedding_norm"] for item in encoder),
                    "fresh_norm_max": max(item["fresh_embedding_norm"] for item in encoder),
                    "cached_norm_min": min(item["cached_embedding_norm"] for item in encoder),
                    "cached_norm_max": max(item["cached_embedding_norm"] for item in encoder)},
                "fresh_vs_cached_sample_quality": diagnostic["aggregate"]})
            print(json.dumps({"dataset": dataset, "mode": mode, "end_to_end_ms": end_to_end["mean"],
                              "sample_quality": diagnostic["aggregate"]}), flush=True)
        del private
    notes = [
        "Real raw-text CPU encoder through TCP Search, protected payload/join decrypt, native rerank, DuetRank and Top20 Fetch/document decrypt. Excludes LLM generation, WAN, setup/model load and connection creation.",
        "Each run has 16 fixed evenly spaced queries, 2 complete warmup passes and 3 complete measurement passes. All 48 measurements retained. Repeated passes use fixed repeat-0 query noise, not independent noise draws.",
        "Nine modes ran once in fixed dataset/mode order, without randomizing order or repeating independent process runs. Warm OS/SQLite caches; do not interpret small differences as stable advantages.",
        "CPU encoder parameters are bfloat16, encode output float32; native semantic rerank is FP32 dot product without re-normalizing stored vectors. Cached-quality and fresh CPU embeddings differ.",
        "Stages overlap: server times are inside RPC spans; never sum all listed stages as end-to-end time. Record lookup is actual per-record SQLite access. No cache/batching optimization was applied during the formal runs.",
        "Actual serialized application bytes include framing; exclude TCP/IP/link overhead. The server uses full FTS/semantic indices, historical workload-union lexical coordinates, and an offline sample-scoped encrypted record store; not a newly built full-corpus encrypted deployment.",
        "The 16-query nDCG diagnostics include calibration/holdout queries and cannot replace full heldout utility evidence. No encoder, retrieval, training or benchmark is rerun by this postprocessor.",
    ]
    output = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "config_sha256": config_sha, "notes": notes,
        "invariant_checks": {"formal_runs": 9, "warmup_observations": 288, "measurement_observations": 432,
            "distinct_query_ids_across_datasets": 48, "all_input_source_hash_sets_identical": True,
            "all_raw_final_timing_and_top20_equal": True, "all_observation_counts_complete": True,
            "failed_or_retried_formal_runs": 0}, "runs": summary_runs}
    (ROOT / "PROTOCOL_TIMING_SUMMARY.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    quality_output = {"schema_version": 1, "scope": notes[-1], "runs": diagnostics}
    (ROOT / "sample_encoder_quality_diagnostic.json").write_text(json.dumps(quality_output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["完整流程正式计时与编码器样本诊断（2026-11-07）", "",
        "9 组全部一次成功完成：每组 16 条固定均匀样本、32 次预热、48 次正式观测；总计 432 次正式观测，无失败或选择快结果重跑。",
        "CPU 单条 BF16 参数编码器（输出 FP32）+ CPU server；两个进程、真实 loopback TCP、真实受保护记录取回/解密、精排、DuetRank、Top20 Fetch 文档解密。",
        "不含模型/进程启动、离线准备、LLM 生成或跨机 WAN。缓存预热且模式固定顺序；小差值不应视为稳定优势。", "",
        "dataset / mode | mean / p50 / p95 (ms) | upload / download (MiB) | record lookup mean (ms, share)"]
    for run in summary_runs:
        e, b = run["end_to_end_ms"], run["application_bytes"]
        lines.append(f"{run['dataset']} / {run['lexical_mode']} | {e['mean']:.3f} / {e['p50']:.3f} / {e['p95']:.3f} | {b['actual_application_upload_bytes']['mean']/2**20:.4f} / {b['actual_application_download_bytes']['mean']/2**20:.4f} | {run['stage_ms']['search_server.record_lookup_ms']['mean']:.3f}, {100*run['record_lookup_fraction_of_total_wall_mean']:.1f}%")
    lines += ["", "仅 16 个 distinct 样本的编码器差异诊断（含 calibration 和 holdout，不能替代全量 heldout 质量）：",
        "dataset / mode | cached nDCG@10 -> fresh nDCG@10 (delta) | Top20 同序样本 /16 | 集合交集均值 /20"]
    for run in summary_runs:
        q = run["fresh_vs_cached_sample_quality"]
        shown_delta = 0.0 if abs(q['paired_delta_ndcg_at_10']) < 1e-12 else q['paired_delta_ndcg_at_10']
        lines.append(f"{run['dataset']} / {run['lexical_mode']} | {q['cached_ndcg_at_10']:.6f} -> {q['fresh_ndcg_at_10']:.6f} ({shown_delta:+.6f}) | {q['queries_top20_exact_on_all_three_passes']}/16 | {q['top20_set_intersection_mean']:.4f}")
    lines += ["", "独立按 qrels 核对：九组各16条样本的Top10 relevance-gain序列全部相同，故nDCG不变；JSON保留均值运算约1e-16的舍入差。",
        "三库这16条样本中的outer-holdout数分别仅7/5/10；不得用这些样本值替代完整heldout评估。",
        "阶段 ms 已完整列在 PROTOCOL_TIMING_SUMMARY.json。RPC 包含 server 子阶段，不能把所有阶段相加。",
        "SQLite 是按记录逐条真实读取；当前缓存/批处理实现会影响这些时延。本轮未实现或测量后续优化。",
        "使用全语料 FTS/semantic 索引、历史 workload candidate-union 词法坐标和离线预备的样本范围密文 records。未声称构建完整语料的新加密部署。",
        "计时重复使用第 0 个 query noise seed；三遍是时钟重复，不是独立噪声质量评估。",
        "缓存 embedding 与 fresh CPU 输出不相同；native semantic 精排为不重归一化的 FP32 点积。",
        "前置 cached-embedding 功能核验为 144/144 通过，但有 39 个 query-cell 局部 CPU/GPU 集合差异，均独立复现原 GPU 边界结果；不得称所有中间数组 bitwise 一致。",
        "原始记录：runs/{dataset}_{mode}/results.json、results.raw.json、client.log、server_requests.jsonl。",
        "详细样本诊断：sample_encoder_quality_diagnostic.json，逐 query 值与来源 SHA256 可复核；未导出私钥或文档文本。"]
    (ROOT / "完整流程计时摘要.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
