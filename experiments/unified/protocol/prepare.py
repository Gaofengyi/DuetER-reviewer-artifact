"""Explicit OFFLINE preparation of sample-scoped protected records.

This command streams corpus data and performs candidate retrieval; the root
runner must schedule it outside concurrent disk/GPU measurements. Online
queries still rerun every index lookup, local scan and ranking stage.
"""
from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

from runtime import initialize
initialize(gpu_dependencies=True)

import numpy as np

from client import QueryOwner, keyed_term, tokenize
from records import DUMMY_ROW, Owner, StoreWriter, pad_lexical_payload
from server import RetrievalIndex


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def run(args):
    started = time.perf_counter()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    fingerprint = hashlib.sha256(config_bytes).hexdigest()
    root = Path(config["original_root"])
    cfg = config["datasets"][args.dataset]
    sys.path.insert(0, str(Path(config["output_root"])))
    import unified_pipeline as pipeline
    print(json.dumps({"stage": "load_query_metadata", "dataset": args.dataset}), flush=True)
    ds = pipeline.prepare_dataset(args.dataset, config=config)
    indices = np.unique(np.linspace(0, len(ds["query_ids"]) - 1, args.query_count, dtype=int))
    queries = [{"index": int(i), "id": ds["query_ids"][i], "text": ds["query_texts"][i],
                "cached_embedding": ds["queries"][i].tolist()} for i in indices]
    if args.output.exists():
        raise FileExistsError("use a new preparation directory; existing stores are immutable")
    args.output.mkdir(parents=True)
    db = sqlite3.connect(f"file:{(root / cfg['fts_database']).as_posix()}?mode=ro", uri=True)
    terms = sorted({keyed_term(token) for query in queries for token in tokenize(query["text"])})
    frequencies = {}
    for first in range(0, len(terms), 800):
        chunk = terms[first:first + 800]
        marks = ",".join("?" for _ in chunk)
        frequencies.update({str(term): int(df) for term, df in db.execute(
            f"SELECT term, doc FROM vocab WHERE term IN ({marks})", chunk)})
    dummy_labels = []
    while len(dummy_labels) < args.term_slots:
        value = os.urandom(16).hex()
        if not db.execute("SELECT 1 FROM vocab WHERE term=?", (value,)).fetchone():
            dummy_labels.append(value)
    db.close()
    lengths_file = root / "cache" / "exact_bm25_client_rerank" / cfg["label"] / "full_doc_lengths_v2.npy"
    lengths = np.load(lengths_file, mmap_mode="r")
    if len(lengths) != cfg["documents"]:
        raise ValueError("document length cache count differs from unified config")
    crypto_owner = Owner.fresh()
    private = {"config_sha256": fingerprint, "keys": crypto_owner.exported(),
        "encoder_path": str(args.encoder_path or root / "cache/models/granite-embedding-small-english-r2"),
        "term_slots": args.term_slots, "document_frequencies": frequencies,
        "dummy_term_labels": dummy_labels, "average_document_length": float(np.mean(lengths)),
        "queries": queries, "token_ids": {}, "document_ids": {},
        "query_view_seed_policy": config["query_seed_policy"],
        "query_noise_repeat": 0, "semantic_payload_dtype": "float32"}
    owner = QueryOwner(config, args.dataset, private, encoder_device=args.encoder_device)
    if args.coverage_encoder:
        print(json.dumps({"stage": "load_coverage_encoder", "device": args.encoder_device}), flush=True)
        owner.load_encoder()
    # Candidate coverage is an offline staging choice, not online ranking.
    # Enlarging only per-cell depth makes the cached-view top-k a subset.
    coverage_config = deepcopy(config)
    coverage_config["datasets"][args.dataset]["semantic_depth"] *= args.semantic_margin
    index = RetrievalIndex(coverage_config, args.dataset)
    print(json.dumps({"stage": "select_sample_records", "queries": len(queries)}), flush=True)
    # Owner-only map. The server sees only a semantic cell-local position.
    semantic_postings = np.load(root / cfg["semantic_cache"] / "semantic_postings.npy", mmap_mode="r")
    frozen_raw = np.load(root / cfg["planner_cache"])[f"budget_{cfg['posting_budget']}"]
    record_keys = set()
    preparation_checks = []
    for query in queries:
        embeddings = [np.asarray(query["cached_embedding"], dtype=np.float32)]
        if args.coverage_encoder:
            embeddings.append(owner.encode_text(query["text"]))
        for variant, embedding in enumerate(embeddings):
            meta, blocks, trace = owner.request(query["text"], query["id"], embedding,
                                                f"prepare-{query['index']}-{variant}")
            selection, counters = index.select(meta, blocks)
            expected = np.asarray(frozen_raw[query["index"], :config["lexical"]["planner_cap"]], dtype="<i8")
            expected = expected[expected >= 0]
            planner_equal = hashlib.sha256(expected.tobytes()).hexdigest() == counters["planner_rows_sha256"]
            if not planner_equal:
                raise AssertionError(f"online FTS5 planner differs from frozen rows for query {query['id']}")
            record_keys.update((path, cell, row) for path, cell, row, _, _ in selection if row >= 0)
            # Prepare ALL planner-1000 lexical records, enabling fair direct
            # controls and coverage independent of their local top-k outcome.
            positions = np.searchsorted(index.candidate_rows, expected)
            for row, cell in zip(expected, index.assignments[positions]):
                record_keys.add((2, int(cell), int(row)))
            preparation_checks.append({"query_id": query["id"], "coverage_embedding_variant": variant,
                "fts5_planner_rows_exactly_equal": planner_equal, "planner_candidates": len(expected),
                "semantic_selected_records": sum(path == 1 and row >= 0 for path, _, row, _, _ in selection)})
        print(json.dumps({"prepared_candidate_query": query["id"], "unique_records": len(record_keys)}), flush=True)
    # Dummy records cover all cells and partitions and have real valid AEAD.
    semantic_cells = {cell for path, cell, _ in record_keys if path == 1}
    # Empty probed cells also need a dummy record; labels are cheap to cover.
    semantic_cells.update(range(len(index.labels)))
    by_row = defaultdict(list)
    for path, cell, lookup_id in record_keys:
        row = int(semantic_postings[int(index.offsets[cell]) + lookup_id]) if path == 1 else lookup_id
        by_row[row].append((path, cell, lookup_id))
    wanted = set(by_row)
    print(json.dumps({"stage": "stream_corpus_encrypt", "records": len(record_keys),
                      "documents": len(wanted)}), flush=True)
    # Candidate scans are complete. Release large local-coordinate mappings
    # and the CPU encoder before streaming the corpus/reading payload vectors.
    index.db.close()
    del index, owner, semantic_postings, frozen_raw, ds
    gc.collect()
    store = StoreWriter(args.output / "records.sqlite3")
    embeddings = np.load(root / cfg["semantic_cache"] / "corpus_embeddings.npy", mmap_mode="r")
    token_ids = private["token_ids"]
    # Full document token sequences are retained, not just query-term TFs.
    for query in queries:
        for term in tokenize(query["text"]):
            if term not in token_ids:
                token_ids[term] = len(token_ids)
    seen = 0
    corpus_bytes_read = 0
    corpus = root / "data" / args.dataset / "corpus.jsonl"
    ids_path = root / cfg["semantic_cache"] / "doc_ids.txt"
    with corpus.open("rb") as corpus_handle, ids_path.open("r", encoding="utf-8") as ids_handle:
        for row, (line, expected_id) in enumerate(zip(corpus_handle, ids_handle)):
            corpus_bytes_read += len(line)
            if row not in wanted:
                continue
            document = json.loads(line)
            if str(document["_id"]) != expected_id.strip():
                raise AssertionError("corpus and frozen embedding row identities differ")
            private["document_ids"][str(row)] = str(document["_id"])
            tokens = tokenize(f"{document.get('title', '')} {document.get('text', '')}".strip())
            if len(tokens) != int(lengths[row]):
                raise AssertionError("full token payload length differs from FTS5 document length")
            for token in tokens:
                if token not in token_ids:
                    token_ids[token] = len(token_ids)
            lexical_payload = pad_lexical_payload(tokens, token_ids)
            semantic_payload = np.asarray(embeddings[row], dtype="<f4").tobytes()
            for path, cell, lookup_id in by_row[row]:
                payload = semantic_payload if path == 1 else lexical_payload
                store.record(path, cell, lookup_id, crypto_owner.make_record(path, cell, row, payload))
            locator = crypto_owner.locator(row)
            store.document(locator, crypto_owner.protect("document", line, locator))
            seen += 1
            if seen % 10000 == 0:
                store.db.commit()
                print(json.dumps({"encrypted_documents": seen, "target_documents": len(wanted)}), flush=True)
            if seen == len(wanted):
                break
    if seen != len(wanted):
        raise ValueError("corpus ended before all selected record documents were prepared")
    for cell in sorted(semantic_cells):
        store.record(1, cell, -1, crypto_owner.make_record(1, cell, DUMMY_ROW,
            bytes(config["semantic"]["encoder_dimension"] * 4)))
    for cell in range(config["lexical"]["partitions"]):
        store.record(2, cell, -1, crypto_owner.make_record(2, cell, DUMMY_ROW, bytes(256)))
    manifest = {"dataset": args.dataset, "config_sha256": fingerprint,
        "stage": "offline sample-scoped AEAD record preparation",
        "sample_query_ids": [q["id"] for q in queries], "sample_query_indices": indices.tolist(),
        "semantic_coverage_margin": args.semantic_margin,
        "coverage_embeddings": "cached plus freshly encoded" if args.coverage_encoder else "cached only",
        "semantic_payload_dtype": "float32", "lexical_payload": "uint32 length and injective owner-assigned uint32 IDs for every token; padded to 256-byte bucket",
        "source_document_count": cfg["documents"], "prepared_documents": seen,
        "protected_path_cell_records": len(record_keys),
        "semantic_records": sum(path == 1 for path, _, _ in record_keys),
        "lexical_records": sum(path == 2 for path, _, _ in record_keys),
        "dummy_records": len(semantic_cells) + config["lexical"]["partitions"],
        "protected_record_bytes": store.payload_bytes, "protected_document_bytes": store.document_bytes,
        "corpus_bytes_streamed": corpus_bytes_read,
        "offline_seconds": time.perf_counter() - started,
        "query_view_seed_policy": config["query_seed_policy"], "repeat": 0,
        "nonce_policy": "fresh independent 256-bit AES keys; one random 32-bit prefix and globally increasing uint64 counter per preparation process; store immutable",
        "index_note": "full existing IVF/FTS5/local-coordinate indices are read online; only protected feature/join/document records are sample-scoped",
        "semantic_lookup_identity": "cell-local position only; global posting-to-document map is owner-only; global join identity is encrypted in capsule",
        "missing_record_policy": "hard error; no online plaintext fallback, lazy encryption or cached ranking replay",
        "cache_note": "offline preparation warms OS/index caches; timing is not a cold-cache claim",
        "preparation_checks": preparation_checks}
    store.finish(manifest)
    private["record_store_sha256"] = file_hash(args.output / "records.sqlite3")
    (args.output / "client_private.json").write_text(json.dumps(private), encoding="utf-8")
    manifest["record_store_sha256"] = private["record_store_sha256"]
    manifest["record_store_file_bytes"] = (args.output / "records.sqlite3").stat().st_size
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("nq", "hotpotqa", "msmarco"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=16)
    parser.add_argument("--semantic-margin", type=int, default=2)
    parser.add_argument("--term-slots", type=int, default=32)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--encoder-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--coverage-encoder", action="store_true")
    run(parser.parse_args())
