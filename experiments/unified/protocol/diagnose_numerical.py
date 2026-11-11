"""Read-only CPU boundary diagnostics; no GPU kernel or performance claim."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from runtime import initialize
initialize(gpu_dependencies=True)
import numpy as np
from client import QueryOwner
from server import RetrievalIndex


def main(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = config["datasets"][args.dataset]
    private = json.loads((args.prepared / "client_private.json").read_text(encoding="utf-8"))
    owner = QueryOwner(config, args.dataset, private, encoder_device="cpu")
    index = RetrievalIndex(config, args.dataset,
        Path(__file__).resolve().parent / "server_indexes" / f"{args.dataset}.sqlite3")
    source = Path(config["original_root"]) / cfg["semantic_cache"]
    postings = np.load(source / "semantic_postings.npy", mmap_mode="r")  # owner-only map
    reference = np.load(Path(config["output_root"]) / "results" / cfg["label"] / "semantic_emitted.npz")
    ref, cells = reference["emitted"], reference["query_cells"]
    global_queries = None
    if args.gpu:
        all_query_ids = reference["query_ids"].tolist()
        cached_queries = np.load(source / cfg["query_embeddings_file"], mmap_mode="r")
        global_queries = owner.pipeline.semantic_global_queries(cached_queries, all_query_ids,
            source / "semantic_dpe_key.npz", repeat=0, config=config)
    previous = json.loads(args.validation.read_text(encoding="utf-8"))
    affected = {row["query_id"] for row in previous["queries"]
                if not row["semantic_cells_and_candidate_sets_exact_match"]}
    output = []
    for query in private["queries"]:
        if query["id"] not in affected:
            continue
        qi = query["index"]
        meta, blocks, _ = owner.request(query["text"], query["id"],
            np.asarray(query["cached_embedding"], dtype=np.float32), f"diagnose-{qi}")
        selection, _ = index.select(meta, blocks)
        views = np.frombuffer(blocks[1], dtype="<f2").reshape(config["semantic"]["probes"], -1).astype(np.float32)
        for probe in range(config["semantic"]["probes"]):
            cell = index.token_to_cell[blocks[0][probe * 16:(probe + 1) * 16]]
            reference_probe = int(np.flatnonzero(cells[qi] == cell)[0])
            expected = set(int(row) for row in ref[0, qi, reference_probe] if row >= 0)
            first, last = int(index.offsets[cell]), int(index.offsets[cell + 1])
            local_docs = np.asarray(postings[first:last], dtype=np.int64)
            positions = [position for path, cc, position, _, _ in selection if path == 1 and cc == cell and position >= 0]
            actual = set(int(local_docs[position]) for position in positions)
            if actual == expected:
                continue
            rows = np.asarray(index.semantic[first:last], dtype=np.float32)
            norms = np.asarray(index.semantic_norms[first:last], dtype=np.float32)
            cpu_scores = 2.0 * (rows @ views[probe]) - norms
            reference_scores64 = 2.0 * (rows.astype(np.float64) @ views[probe].astype(np.float64)) - norms.astype(np.float64)
            cutoff = float(min(cpu_scores[position] for position in positions))
            gpu_diagnostic = None
            gpu_scores = None
            if args.gpu:
                import torch
                # Reproduce the quality runner's original cell qi collection
                # AND its batch-128 slice, not a different single-query kernel.
                query_locations = np.flatnonzero(np.any(cells == cell, axis=1))
                position_in_cell = int(np.flatnonzero(query_locations == qi)[0])
                batch_size = config["runtime"]["query_batch_size"]
                batch_start = (position_in_cell // batch_size) * batch_size
                batch_end = min(batch_start + batch_size, len(query_locations))
                transform = owner.pipeline.cell_transform(config["semantic"]["key_label"].encode(), cell,
                    config["semantic"]["work_dimension"], config["semantic"]["local_dimension"])
                original_views = owner.pipeline.apply_transform(global_queries[query_locations], transform).astype(np.float16).astype(np.float32)
                query_bytes_equal = np.array_equal(original_views[position_in_cell].astype(np.float16).view(np.uint16),
                                                    views[probe].astype(np.float16).view(np.uint16))
                with torch.inference_mode():
                    rows_gpu = torch.from_numpy(np.array(rows, dtype=np.float32)).cuda()
                    norms_gpu = torch.from_numpy(np.array(norms, dtype=np.float32)).cuda()
                    q_gpu = torch.from_numpy(original_views[batch_start:batch_end]).cuda()
                    scores_gpu = 2.0 * (q_gpu @ rows_gpu.T) - norms_gpu[None, :]
                    chosen_gpu = torch.topk(scores_gpu, k=cfg["semantic_depth"], dim=1).indices.cpu().numpy()
                    gpu_scores = scores_gpu[position_in_cell - batch_start].cpu().numpy()
                    gpu_positions = chosen_gpu[position_in_cell - batch_start]
                    torch.cuda.synchronize()
                original_expected = ref[0, qi, reference_probe]
                gpu_diagnostic = {"local_query_fp16_bytes_identical": bool(query_bytes_equal),
                    "original_gpu_ordered_selection_reproduced": bool(np.array_equal(local_docs[gpu_positions], original_expected)),
                    "original_cell_query_count": len(query_locations), "batch_start": batch_start,
                    "batch_end": batch_end, "query_position_in_cell": position_in_cell,
                    "batch_query_indices": query_locations[batch_start:batch_end].tolist(),
                    "gpu_cutoff": float(np.min(gpu_scores[gpu_positions])),
                    "gpu_dtype": str(scores_gpu.dtype), "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
                    "max_abs_gpu_minus_cpu_scores": float(np.max(np.abs(gpu_scores - cpu_scores)))}
                del rows_gpu, norms_gpu, q_gpu, scores_gpu
            changed = []
            for doc in sorted(actual ^ expected):
                pos = int(np.flatnonzero(local_docs == doc)[0])
                detail = {"doc_row": doc, "local_position": pos,
                    "side": "cpu_only" if doc in actual else "gpu_quality_only",
                    "cpu_fp32_score": float(cpu_scores[pos]),
                    "same_stored_inputs_fp64_score": float(reference_scores64[pos]),
                    "cpu_score_minus_cpu_cutoff": float(cpu_scores[pos]) - cutoff}
                if gpu_scores is not None:
                    detail["gpu_fp32_score"] = float(gpu_scores[pos])
                changed.append(detail)
            output.append({"query_id": query["id"], "cell": cell, "posting_count": len(rows),
                "local_depth": cfg["semantic_depth"], "cpu_cutoff": cutoff,
                "changed_records": changed,
                "gpu_reproduction": gpu_diagnostic,
                "all_changed_records_tied_at_cpu_cutoff": all(item["cpu_score_minus_cpu_cutoff"] == 0 for item in changed),
                "max_abs_cpu_score_delta_from_cutoff": max(abs(item["cpu_score_minus_cpu_cutoff"]) for item in changed)})
    report = {"dataset": args.dataset, "scope": ("Original batch-shape GPU FP32 reproduction plus CPU FP32/FP64 on identical stored FP16 values"
              if args.gpu else "CPU FP32 and FP64 arithmetic on identical stored FP16 rows and query views; GPU not rerun"),
              "mismatched_cells": output}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", choices=("nq", "hotpotqa", "msmarco"), required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", action="store_true")
    main(p.parse_args())
