"""Independent loopback retrieval server. Does not load owner keys/encoder.

All FTS5 planning, local-coordinate scans/top-k and protected-record lookups
run anew for every Search RPC. Semantic identities are cell-local positions;
the server does not load the owner-side global semantic posting map.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
import time

from runtime import initialize
initialize()

import numpy as np

from records import StoreReader
from wire import decode, encode, receive, send


def top_indices(scores, depth):
    if depth <= 0:
        return np.empty(0, dtype=np.int64)
    if depth >= len(scores):
        return np.argsort(-scores, kind="stable")
    chosen = np.argpartition(-scores, depth - 1)[:depth]
    return chosen[np.argsort(-scores[chosen], kind="stable")]


class RetrievalIndex:
    def __init__(self, config: dict, dataset: str, server_index: Path | None = None):
        self.config, self.name = config, dataset
        self.cfg = config["datasets"][dataset]
        root = Path(config["original_root"])
        sem = root / self.cfg["semantic_cache"]
        local = root / self.cfg["semantic_local_cache"]
        lex = root / self.cfg["lexical_local_cache"]
        source = root / self.cfg["lexical_source"]
        self.labels = np.load(sem / "semantic_labels.npy")
        self.token_to_cell = {bytes(label): cell for cell, label in enumerate(self.labels)}
        self.offsets = np.load(sem / "semantic_offsets.npy", mmap_mode="r")
        self.semantic = np.load(local / "semantic_compartment_fp16.npy", mmap_mode="r")
        self.semantic_norms = np.load(local / "semantic_compartment_norms.npy", mmap_mode="r")
        self.candidate_rows = np.load(source / "candidate_rows.npy", mmap_mode="r")
        self.assignments = np.load(lex / "lexical_compartment_cells.npy", mmap_mode="r")
        self.lexical = np.load(lex / "lexical_compartment_fp16.npy", mmap_mode="r")
        self.lexical_norms = np.load(lex / "lexical_compartment_norms.npy", mmap_mode="r")
        self.database_path = server_index or root / self.cfg['fts_database']
        self.db = sqlite3.connect(f"file:{self.database_path.as_posix()}?mode=ro", uri=True)
        self.db.execute("PRAGMA query_only=ON")
        self.db.execute("PRAGMA cache_size=-262144")
        if server_index is not None and self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='docmap'").fetchone():
            raise ValueError("server index still contains owner docmap identities")
        expected = (int(self.offsets[-1]), config["semantic"]["local_dimension"])
        if self.semantic.shape != expected:
            raise ValueError(f"semantic cache shape {self.semantic.shape} != {expected}")
        if self.lexical.shape != (len(self.candidate_rows), config["lexical"]["local_dimension"]):
            raise ValueError("lexical cache shape mismatch")

    def select(self, meta: dict, blocks: list[bytes]):
        started = time.perf_counter_ns()
        semcfg, lexcfg = self.config["semantic"], self.config["lexical"]
        probes, parts = semcfg["probes"], lexcfg["partitions"]
        if not meta.get("semantic_enabled", True):
            probes = 0
        lexical_mode = meta.get("lexical_mode", "ccadpe")
        if lexical_mode not in ("ccadpe", "direct300", "direct1000"):
            raise ValueError("unknown lexical mode")
        if meta.get("dataset") != self.name or len(blocks) != 4:
            raise ValueError("invalid Search dataset or blocks")
        labels, sem_bytes, term_bytes, lex_bytes = blocks
        if len(labels) != probes * 16 or len(term_bytes) != meta["term_slots"] * 16:
            raise ValueError("incorrect fixed label slots")
        sem_queries = np.frombuffer(sem_bytes, dtype="<f2").reshape(probes, semcfg["local_dimension"]).astype(np.float32)
        lex_queries = np.frombuffer(lex_bytes, dtype="<f2").reshape(
            parts if lexical_mode == "ccadpe" else 0, lexcfg["local_dimension"]).astype(np.float32)
        if not (np.isfinite(sem_queries).all() and np.isfinite(lex_queries).all()):
            raise ValueError("non-finite query coordinates")
        decoded = time.perf_counter_ns()
        result = []
        reads = 0
        # Each tuple is (path, internal cell, local lookup ID, planner position,
        # response group). Every group has a constant number of output slots.
        # Semantic lookup IDs are cell-local positions, NEVER global doc rows.
        for probe in range(probes):
            token = labels[probe * 16:(probe + 1) * 16]
            if token not in self.token_to_cell:
                raise ValueError("unknown semantic probe")
            cell = self.token_to_cell[token]
            first, last = int(self.offsets[cell]), int(self.offsets[cell + 1])
            local = np.asarray(self.semantic[first:last], dtype=np.float32)
            scores = 2.0 * (local @ sem_queries[probe]) - self.semantic_norms[first:last]
            positions = top_indices(scores, min(self.cfg["semantic_depth"], len(local)))
            result.extend((1, cell, int(position), -1, probe) for position in positions)
            result.extend((1, cell, -1, -1, probe)
                          for _ in range(self.cfg["semantic_depth"] - len(positions)))
            reads += last - first
        semantic_done = time.perf_counter_ns()
        terms = [term_bytes[i:i + 16].hex() for i in range(0, len(term_bytes), 16)]
        # Padding labels are looked up too, so the request has a fixed shape.
        rows = self.db.execute(
            "SELECT rowid FROM postings WHERE postings MATCH ? ORDER BY bm25(postings) LIMIT ?",
            (" OR ".join(terms), lexcfg["planner_cap"])).fetchall()
        candidates = np.asarray([int(item[0]) - 1 for item in rows], dtype=np.int64)
        planner_digest = hashlib.sha256(candidates.astype("<i8").tobytes()).hexdigest()
        planner_done = time.perf_counter_ns()
        positions = np.searchsorted(self.candidate_rows, candidates)
        if len(positions) and (np.any(positions >= len(self.candidate_rows)) or
                               np.any(self.candidate_rows[positions] != candidates)):
            raise RuntimeError("online FTS5 candidate absent from scoped lexical coordinate cache")
        cells = np.asarray(self.assignments[positions], dtype=np.int32)
        if lexical_mode != "ccadpe":
            cap = 300 if lexical_mode == "direct300" else 1000
            result.extend((2, int(cells[o]), int(row), o, 0) for o, row in enumerate(candidates[:cap]))
            result.extend((2, 0, -1, -1, 0) for _ in range(cap - min(cap, len(candidates))))
            finished = time.perf_counter_ns()
            return result, {
                "request_decode_vectors_ms": (decoded - started) / 1e6,
                "semantic_lookup_refine_ms": (semantic_done - decoded) / 1e6,
                "lexical_fts5_planner_ms": (planner_done - semantic_done) / 1e6,
                "lexical_partition_refine_ms": 0.0,
                "direct_candidate_selection_ms": (finished - planner_done) / 1e6,
                "semantic_posting_reads": reads, "planner_candidates": len(candidates),
                "planner_rows_sha256": planner_digest}
        for cell in range(parts):
            ordinals = np.flatnonzero(cells == cell)
            row_positions = positions[ordinals]
            local = np.asarray(self.lexical[row_positions], dtype=np.float32)
            scores = 2.0 * (local @ lex_queries[cell]) - self.lexical_norms[row_positions]
            chosen = top_indices(scores, min(lexcfg["local_depth"], len(scores)))
            selected = ordinals[chosen]
            result.extend((2, cell, int(candidates[o]), int(o), cell) for o in selected)
            result.extend((2, cell, -1, -1, cell)
                          for _ in range(lexcfg["local_depth"] - len(selected)))
        finished = time.perf_counter_ns()
        return result, {
            "request_decode_vectors_ms": (decoded - started) / 1e6,
            "semantic_lookup_refine_ms": (semantic_done - decoded) / 1e6,
            "lexical_fts5_planner_ms": (planner_done - semantic_done) / 1e6,
            "lexical_partition_refine_ms": (finished - planner_done) / 1e6,
            "semantic_posting_reads": reads,
            "planner_candidates": len(candidates),
            "planner_rows_sha256": planner_digest,
        }


def run(args):
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    server_index = args.server_index or Path(__file__).resolve().parent / "server_indexes" / f"{args.dataset}.sqlite3"
    if not server_index.exists():
        raise FileNotFoundError("isolated server index is required; no original owner-index fallback")
    index = RetrievalIndex(config, args.dataset, server_index)
    index_hash = hashlib.sha256()
    with server_index.open("rb") as handle:
        for block in iter(lambda: handle.read(2 ** 20), b""):
            index_hash.update(block)
    store = StoreReader(args.prepared / "records.sqlite3")
    fingerprint = hashlib.sha256(config_bytes).hexdigest()
    if store.manifest["config_sha256"] != fingerprint:
        raise ValueError("prepared records use a different unified config")
    request_log = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", args.port))
        listener.listen(1)
        ready = {"host": "127.0.0.1", "port": listener.getsockname()[1], "pid": os.getpid(),
                 "config_sha256": fingerprint, "server_backend": "CPU FP32 per-query local scans",
                 "server_index": str(server_index), "server_index_sha256": index_hash.hexdigest(),
                 "server_index_bytes": server_index.stat().st_size,
                 "lexical_coordinate_rows": len(index.candidate_rows),
                 "corpus_documents": index.cfg["documents"],
                 "lexical_coordinate_scope": "existing benchmark-workload candidate-union cache; missing online candidate is a hard error"}
        args.ready.write_text(json.dumps(ready, indent=2), encoding="utf-8")
        print(json.dumps(ready), flush=True)
        while True:
            sock, _ = listener.accept()
            with sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while True:
                    try:
                        body, received = receive(sock)
                    except EOFError:
                        break
                    begin = time.perf_counter_ns()
                    try:
                        meta, blocks = decode(body)
                        parsed = time.perf_counter_ns()
                        if meta["kind"] == "search":
                            selection, counters = index.select(meta, blocks)
                            lookup_start = time.perf_counter_ns()
                            records = [store.record(path, cell, row) for path, cell, row, _, _ in selection]
                            lookup_done = time.perf_counter_ns()
                            out = {"kind": "search_result", "slots": [[path, group, order]
                                   for path, _, _, order, group in selection],
                                   "server": counters | {
                                       "request_parse_ms": (parsed - begin) / 1e6,
                                       "record_lookup_ms": (lookup_done - lookup_start) / 1e6,
                                       "request_application_bytes": received}}
                        elif meta["kind"] == "fetch":
                            if len(blocks) != 1 or len(blocks[0]) != meta["count"] * 8 or meta["count"] > 20:
                                raise ValueError("invalid Top20 Fetch request")
                            lookup_start = time.perf_counter_ns()
                            records = [store.document(blocks[0][i:i + 8]) for i in range(0, len(blocks[0]), 8)]
                            lookup_done = time.perf_counter_ns()
                            out = {"kind": "fetch_result", "count": len(records), "server": {
                                "request_parse_ms": (parsed - begin) / 1e6,
                                "document_lookup_ms": (lookup_done - lookup_start) / 1e6,
                                "request_application_bytes": received}}
                        elif meta["kind"] == "ping":
                            out, records = {"kind": "pong", **ready}, []
                        elif meta["kind"] == "shutdown":
                            if args.log:
                                args.log.write_text("".join(json.dumps(item) + "\n" for item in request_log), encoding="utf-8")
                            send(sock, encode({"kind": "stopped"}, []))
                            return
                        else:
                            raise ValueError("unsupported RPC")
                        serialize_start = time.perf_counter_ns()
                        response = encode(out, records)
                        serialize_done = time.perf_counter_ns()
                        # Serialization is measured locally in a side-channel log:
                        # injecting the time would itself require a second encoding.
                        sent = send(sock, response)
                        ended = time.perf_counter_ns()
                        if args.log:
                            request_log.append({"kind": meta["kind"], "request_id": meta.get("request_id"),
                                "serialize_ms": (serialize_done - serialize_start) / 1e6,
                                "send_ms": (ended - serialize_done) / 1e6,
                                "server_wall_after_receive_ms": (ended - begin) / 1e6,
                                "response_application_bytes": sent})
                    except Exception as exc:
                        send(sock, encode({"error": f"{type(exc).__name__}: {exc}"}, []))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("nq", "hotpotqa", "msmarco"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--server-index", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    run(parser.parse_args())
