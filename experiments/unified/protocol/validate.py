"""Algorithm/record consistency checks, explicitly NOT timing evidence.

Uses cached embeddings to isolate the protocol implementation from fresh
single-query encoder differences. Timed client.py always calls the encoder.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from runtime import initialize
initialize(gpu_dependencies=True)
import numpy as np

from client import QueryOwner
from records import StoreReader
from server import RetrievalIndex
from wire import decode, encode


def run(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = config["datasets"][args.dataset]
    private = json.loads((args.prepared / "client_private.json").read_text(encoding="utf-8"))
    if private["config_sha256"] != hashlib.sha256(args.config.read_bytes()).hexdigest():
        raise ValueError("config changed after record preparation")
    owner = QueryOwner(config, args.dataset, private, encoder_device="cpu")
    owner.load_ranker(args.fusion_model)
    fusion_report = json.loads(args.fusion_report.read_text(encoding="utf-8"))
    backoff = float(fusion_report["selection"]["semantic_backoff_weight"])
    server_index = args.server_index or Path(__file__).resolve().parent / "server_indexes" / f"{args.dataset}.sqlite3"
    if not server_index.exists():
        raise FileNotFoundError("validation requires the isolated server index")
    index = RetrievalIndex(config, args.dataset, server_index)
    store = StoreReader(args.prepared / "records.sqlite3")
    reference = args.reference_dir or Path(config["output_root"]) / "results" / cfg["label"]
    sem = np.load(reference / "semantic.rankings.npz")["semantic"]
    emitted_file = np.load(reference / "semantic_emitted.npz")
    emitted, reference_cells = emitted_file["emitted"], emitted_file["query_cells"]
    lex = np.load(args.reference_lexical or reference / "exact_bm25.rankings.npz")["lexical_dpe"]
    fused = np.load(args.reference_fused or reference / "duetrank.rankings.npz")["dual"]
    frozen = np.load(Path(config["original_root"]) / cfg["planner_cache"])[f"budget_{cfg['posting_budget']}"]
    candidate_file = reference / "lexical_candidates.npz"
    candidates = np.load(candidate_file)["lexical_dpe"] if args.lexical_mode == "ccadpe" else None
    results = []
    for query in private["queries"]:
        qi = query["index"]
        embedding = np.asarray(query["cached_embedding"], dtype=np.float32)
        meta, blocks, trace = owner.request(query["text"], query["id"], embedding,
            f"validate-{qi}", args.lexical_mode, True)
        # Exercise real encoding/decoding without claiming network or timing.
        decoded_meta, decoded_blocks = decode(encode(meta, blocks))
        selection, stats = index.select(decoded_meta, decoded_blocks)
        raw = np.asarray(frozen[qi, :config["lexical"]["planner_cap"]], dtype="<i8")
        raw = raw[raw >= 0]
        planner_match = hashlib.sha256(raw.tobytes()).hexdigest() == stats["planner_rows_sha256"]
        metadata = {"slots": [[path, group, ordinal] for path, _, _, ordinal, group in selection]}
        record_blocks = [store.record(path, cell, row) for path, cell, row, _, _ in selection]
        response_meta, records = decode(encode(metadata, record_blocks))
        top20, locators, rerank = owner.rerank(query["text"], embedding, response_meta, records, trace,
                                              backoff, args.lexical_mode, True)
        expected_cells = {int(cell): sorted(int(row) for row in emitted[0, qi, slot] if row >= 0)
                          for slot, cell in enumerate(reference_cells[qi])}
        got_cells = {}
        for (path, cell, lookup_id, _, _), protected_record in zip(selection, records):
            if path == 1:
                got_cells.setdefault(cell, [])
                row, _, version, _ = owner.owner.open_record(protected_record)
                if version:
                    got_cells[cell].append(row)
        got_cells = {cell: sorted(rows) for cell, rows in got_cells.items()}
        lex_input = [row for _, _, row, _, _ in sorted(
            (item for item in selection if item[0] == 2 and item[2] >= 0), key=lambda item: item[3])]
        cap = 1000 if args.lexical_mode == "direct1000" else 300
        lex_input = lex_input[:cap]
        expected_input = (candidates[0, qi][candidates[0, qi] >= 0].tolist()
                          if candidates is not None else raw[:cap].tolist())
        semantic_rank_match = rerank["semantic_rank100"] == sem[0, qi].tolist()
        lexical_rank_match = rerank["lexical_rank"] == lex[0, qi, :cap].tolist()
        final_match = top20 == [int(row) for row in fused[0, qi, :20] if row >= 0]
        # AEAD document fetch integrity is also checked locally here; the timed
        # run performs these store accesses through the separate server RPC.
        for row, locator in zip(top20, locators):
            document = json.loads(owner.owner.open("document", store.document(locator), locator))
            if str(document["_id"]) != private["document_ids"][str(row)]:
                raise AssertionError("document AEAD/identity validation failed")
        result = {"query_id": query["id"], "query_index": qi,
            "planner_rows_exact_match": planner_match,
            "semantic_routing_cells_match": set(got_cells) == set(expected_cells),
            "semantic_cells_and_candidate_sets_exact_match": got_cells == expected_cells,
            "lexical_precap_survivors_capped_in_planner_order_match": lex_input == expected_input,
            "semantic_rank100_exact_match": semantic_rank_match,
            "lexical_rank_exact_match": lexical_rank_match,
            "duetrank_top20_exact_match": final_match,
            "payload_join_document_aead_and_identity_valid": True}
        result["diagnostics"] = {
            "semantic_added_cells": sorted(set(got_cells) - set(expected_cells)),
            "semantic_missing_cells": sorted(set(expected_cells) - set(got_cells)),
            "semantic_mismatched_cell_count": sum(got_cells.get(cell) != expected_cells.get(cell)
                                                   for cell in set(got_cells) | set(expected_cells)),
            "top20_actual": top20, "top20_reference": fused[0, qi, :20].tolist()}
        result["all_match"] = all(value for key, value in result.items()
                                  if key.endswith("match") or key.endswith("valid"))
        result["functional_match"] = all(result[key] for key in (
            "planner_rows_exact_match", "semantic_routing_cells_match",
            "lexical_precap_survivors_capped_in_planner_order_match",
            "semantic_rank100_exact_match", "lexical_rank_exact_match",
            "duetrank_top20_exact_match", "payload_join_document_aead_and_identity_valid"))
        results.append(result)
        print(json.dumps({"query_id": query["id"], "all_match": result["all_match"],
            "functional_match": result["functional_match"],
            "mismatched_semantic_cells": result["diagnostics"]["semantic_mismatched_cell_count"],
            "failed_checks": [key for key, value in result.items()
                              if (key.endswith("match") or key.endswith("valid")) and value is False]}), flush=True)
    report = {"scope": "cached-embedding algorithm and AEAD consistency only; no latency claim",
        "dataset": args.dataset, "lexical_mode": args.lexical_mode,
        "server_index": str(server_index),
        "reference_directory": str(reference), "config_sha256": private["config_sha256"],
        "all_queries_match": all(item["all_match"] for item in results), "queries": results,
        "all_functional_checks_match": all(item["functional_match"] for item in results),
        "local_backend_difference_policy": "permitted only for local candidate sets when routing, both final native rankings, fusion Top20 and all structural/AEAD checks match exactly"
            if args.allow_local_backend_differences else "strict all-intermediate equality required",
        "backend_note": "Protocol server is FP32 CPU per-query; quality semantic local scores may use FP32 GPU batches. Any differing cells/candidates/ranks are reported, never silently accepted as exact.",
        "encoder_note": "Fresh online encoder differences are reported by client.py after its measured wall-clock interval; this check intentionally uses cached embeddings."}
    report["accepted_for_timing"] = (report["all_queries_match"] or
        (args.allow_local_backend_differences and report["all_functional_checks_match"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["accepted_for_timing"]:
        raise SystemExit("consistency mismatch: inspect report before presenting timing as matched")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("nq", "hotpotqa", "msmarco"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--server-index", type=Path)
    parser.add_argument("--fusion-model", type=Path, required=True)
    parser.add_argument("--fusion-report", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--reference-lexical", type=Path)
    parser.add_argument("--reference-fused", type=Path)
    parser.add_argument("--lexical-mode", choices=("ccadpe", "direct300", "direct1000"), default="ccadpe")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-local-backend-differences", action="store_true")
    run(parser.parse_args())
