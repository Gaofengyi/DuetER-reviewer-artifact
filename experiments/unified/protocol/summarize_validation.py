"""Small-file provenance summary; never starts a retrieval or timing run."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent
rows = []
for dataset in ("nq", "hotpotqa", "msmarco"):
    prepared = json.loads((root / "prepared" / f"{dataset}_cpu_v1" / "manifest.json").read_text())
    reports = [json.loads((root / "validation/final_v2" / f"{dataset}_{mode}.json").read_text())
               for mode in ("ccadpe", "direct300", "direct1000")]
    gpu_path = (root / "validation/nq_v1/local_boundary_gpu_diagnostic.json" if dataset == "nq"
                else root / "validation/final_v2" / f"{dataset}_boundary_gpu.json")
    gpu = json.loads(gpu_path.read_text())["mismatched_cells"]
    rows.append({"dataset": dataset, "sample_queries": 16,
        "prepared_documents": prepared["prepared_documents"],
        "prepared_path_cell_records_excluding_dummies": prepared["protected_path_cell_records"],
        "prepared_dummy_records": prepared["dummy_records"],
        "store_file_bytes": prepared["record_store_file_bytes"],
        "protected_record_bytes_including_dummies": prepared["protected_record_bytes"],
        "protected_document_bytes": prepared["protected_document_bytes"],
        "functional_mode_query_checks": sum(len(report["queries"]) for report in reports),
        "functional_mode_query_checks_passed": sum(sum(row["functional_match"] for row in report["queries"]) for report in reports),
        "queries_with_all_local_sets_exact": sum(row["semantic_cells_and_candidate_sets_exact_match"] for row in reports[0]["queries"]),
        "query_cell_instances": 16 * 128, "different_query_cell_instances": len(gpu),
        "exchanged_local_records": sum(len(cell["changed_records"]) // 2 for cell in gpu),
        "all_diagnostic_query_bytes_equal": all(cell["gpu_reproduction"]["local_query_fp16_bytes_identical"] for cell in gpu),
        "all_original_ordered_gpu_selections_reproduced": all(cell["gpu_reproduction"]["original_gpu_ordered_selection_reproduced"] for cell in gpu),
        "max_absolute_gpu_cpu_score_difference": max(cell["gpu_reproduction"]["max_abs_gpu_minus_cpu_scores"] for cell in gpu),
        "boundary_diagnostic": str(gpu_path)})
output = {"status": "functional validation complete; formal end-to-end timing pending resource handoff",
    "config_sha256": prepared["config_sha256"], "datasets": rows,
    "scope": "48 distinct sampled queries, each checked under 3 modes (144 comparisons); not 144 independent queries",
    "accepted_difference": "Stored local-query bytes are identical; original GPU batch selections reproduced. CPU/GPU FP32 local-cutoff sets differ in 39 query-cell instances. All native semantic100, lexical rankings, fused Top20, planner order, routing, and AEAD/identity checks match exactly for these sampled cached-embedding cases.",
    "not_claimed": "Not bitwise equality of all intermediate local sets; not yet fresh-CPU-encoder equality; not a new full-corpus encrypted record-store build."}
(root / "PROTOCOL_VALIDATION_SUMMARY.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
lines = ["协议功能核对完成；正式端到端计时尚待资源交接。", "",
         "48条独立样本query、每条3种流程，共144次功能比较全部通过。",
         "FTS原序列、路由、native semantic100、词法精排、融合Top20和AEAD/文档身份均完全一致。",
         "不能称所有中间候选逐位一致：NQ/Hot/MS分别有3/13/23个query-cell边界集合不同（每库共2048个query-cell）。",
         "已对全部39个差异实例重现原GPU batch边界：query FP16字节全同，原GPU有序选择全复现。",
         "CPU/GPU FP32累加差异最高0.00067138671875；不通过修改quality排名消除这些原始差异。",
         "另一次semantic平分排序实现差异已修复；旧nq_v1报告保留，正式核对见validation/final_v2。", "",
         "样本密文准备："]
for row in rows:
    lines.append(f"{row['dataset']}: {row['prepared_documents']:,}篇文档；{row['prepared_path_cell_records_excluding_dummies']:,}条path-cell记录（另有dummy）；store {row['store_file_bytes']:,} bytes。")
lines += ["", "完整FTS5/semantic IVF用于在线检索；词法坐标复用工作负载candidate-union cache，AEAD记录仅为样本需要的范围。",
          "线上不回放排名，缺记录硬失败；owner完整token序列和FP32语义payload经实际AES-GCM保护。",
          "正式时钟将每次实际执行CPU query encoder、loopback TCP两进程Search/Fetch及文档解密；不含LLM/WAN。",
          "fresh CPU encoder与cached quality浮点实例的范数/输出/排名差异将在正式运行后另报，不能由当前cached核对替代。"]
(root / "协议核对摘要.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"functional_checks_passed": sum(row["functional_mode_query_checks_passed"] for row in rows),
                  "different_query_cell_instances": sum(row["different_query_cell_instances"] for row in rows)}))
