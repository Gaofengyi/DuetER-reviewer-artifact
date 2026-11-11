"""Owner-side online query pipeline and real Search/Fetch RPC measurement.

Encoder/model construction and immutable setup are outside measured queries.
Every measured query starts from its text and ends after document decryption.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time

from runtime import initialize
initialize(gpu_dependencies=True)

import joblib
import numpy as np

from records import Owner, open_lexical_payload
from wire import rpc

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
LEXICAL_KEY = b"DuetDPE-keyed-vocabulary-tail-v2"


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def keyed_term(text):
    import hmac
    return hmac.new(LEXICAL_KEY, text.encode(), hashlib.sha256).hexdigest()[:32]


def choose_terms(frequencies, budget, documents, alpha=0.5):
    ranked = sorted(frequencies, key=lambda term: (
        -math.log1p(documents / max(frequencies[term], 1)) / frequencies[term] ** alpha,
        frequencies[term], term))
    selected, spent = [], 0
    for term in ranked:
        if not selected or spent + frequencies[term] <= budget:
            selected.append(term)
            spent += frequencies[term]
    return selected


class QueryOwner:
    def __init__(self, config: dict, dataset: str, private: dict, *, encoder_device: str):
        self.config, self.dataset, self.private = config, dataset, private
        self.cfg = config["datasets"][dataset]
        root = Path(config["original_root"])
        sem = root / self.cfg["semantic_cache"]
        self.semantic_source = sem
        sys.path.insert(0, str(Path(config["output_root"])))
        import unified_pipeline
        self.pipeline = unified_pipeline
        self.mean = np.load(sem / "semantic_mean.npy")
        self.centroids = np.load(sem / "semantic_centroids.npy")
        self.labels = np.load(sem / "semantic_labels.npy")
        self.owner = Owner(private["keys"])
        self.device = encoder_device
        self.encoder = None
        self.model = None

    def load_encoder(self):
        from sentence_transformers import SentenceTransformer
        model_path = self.private["encoder_path"]
        self.encoder = SentenceTransformer(model_path, device=self.device, local_files_only=True)
        self.encoder_parameter_dtype = str(next(self.encoder.parameters()).dtype)
        print(json.dumps({"encoder_loaded": model_path, "device": self.device,
                          "parameter_dtype": self.encoder_parameter_dtype,
                          "local_files_only": True}), flush=True)
        if self.device.startswith("cuda"):
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def load_ranker(self, model_path: Path):
        self.model = joblib.load(model_path)
        # Keep the established feature functions; never train in this harness.
        sys.path.insert(0, str(Path(self.config["original_root"])))
        from experiment_adaptive_fusion import query_features
        from experiment_candidate_rank_fusion import learned_rankings, mix_rankings
        self.query_features = query_features
        self.learned_rankings = learned_rankings
        self.mix_rankings = mix_rankings

    def encode_text(self, text):
        if self.encoder is None:
            raise RuntimeError("encoder has not been loaded")
        encoded = self.encoder.encode([text], batch_size=1, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True)
        self.last_encoder_output_dtype = str(encoded.dtype)
        embedding = encoded.astype(np.float32)[0]
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()
        return embedding

    def request(self, text: str, query_id: str, embedding: np.ndarray | None, request_id: str,
                lexical_mode="ccadpe", semantic_enabled=True):
        started = time.perf_counter_ns()
        raw_terms = list(dict.fromkeys(tokenize(text)))
        tokens = {term: keyed_term(term) for term in raw_terms}
        frequencies = {token: int(self.private["document_frequencies"][token])
                      for token in tokens.values() if token in self.private["document_frequencies"]
                      and self.private["document_frequencies"][token] > 0}
        selected = sorted(choose_terms(frequencies, self.cfg["posting_budget"], self.cfg["documents"]))
        term_slots = self.private["term_slots"]
        if len(selected) > term_slots:
            raise ValueError("query exceeds fixed lexical label slot budget")
        # Fixed dummy labels are guaranteed absent at offline preparation.
        padding = self.private["dummy_term_labels"][:term_slots - len(selected)]
        lexical_labels = b"".join(bytes.fromhex(token) for token in selected + padding)
        probes = self.config["semantic"]["probes"]
        cells = []
        semantic = np.empty((0, self.config["semantic"]["local_dimension"]), dtype=np.float16)
        if semantic_enabled:
            residual = embedding - self.mean
            residual /= max(float(np.linalg.norm(residual)), 1e-12)
            scores = self.centroids @ residual
            cells = np.argpartition(-scores, probes - 1)[:probes]
            sem_base = self.pipeline.semantic_global_queries(embedding[None, :], [query_id],
                self.semantic_source / "semantic_dpe_key.npz", repeat=0, config=self.config)[0]
            semantic = self.pipeline.semantic_query_views(sem_base, cells, config=self.config)
        lexical = np.empty((0, self.config["lexical"]["local_dimension"]), dtype=np.float16)
        if lexical_mode == "ccadpe":
            vector = self.pipeline.query_mips(text, frequencies, self.cfg["documents"],
                                              self.config["lexical"]["hash_dimension"])
            lexical = self.pipeline.lexical_query_views(vector[None], [query_id], repeat=0, config=self.config)[0]
        blocks = [b"".join(bytes(self.labels[cell]) for cell in cells),
                  np.asarray(semantic, dtype="<f2").tobytes(), lexical_labels,
                  np.asarray(lexical, dtype="<f2").tobytes()]
        meta = {"kind": "search", "dataset": self.dataset, "request_id": request_id,
                "term_slots": term_slots, "lexical_mode": lexical_mode,
                "semantic_enabled": semantic_enabled}
        return meta, blocks, {"selected_terms": len(selected),
                             "estimated_postings": sum(frequencies[term] for term in selected),
                             "semantic_query_seed": self.pipeline.query_seed(query_id, "semantic", 0, self.config),
                             "lexical_query_seed": self.pipeline.query_seed(query_id, "lexical", 0, self.config),
                             "request_build_ms": (time.perf_counter_ns() - started) / 1e6}

    def rerank(self, text, embedding, response_meta, records, trace, backoff,
               lexical_mode="ccadpe", semantic_enabled=True):
        started = time.perf_counter_ns()
        if len(records) != len(response_meta["slots"]):
            raise ValueError("response slot count mismatch")
        semantic, lexical, locators = {}, {}, {}
        dummy_count = 0
        for (path, _, planner_position), record in zip(response_meta["slots"], records):
            row, locator, version, payload = self.owner.open_record(record)
            if version == 0:
                dummy_count += 1
                continue
            if row in locators and locators[row] != locator:
                raise ValueError("inconsistent join locator")
            locators[row] = locator
            if path == 1:
                vector = np.frombuffer(payload, dtype="<f4")
                if len(vector) != self.config["semantic"]["encoder_dimension"]:
                    raise ValueError("semantic payload dimension mismatch")
                semantic.setdefault(row, vector)
            elif path == 2:
                if row in lexical:
                    raise ValueError("lexical row returned from multiple partitions")
                lexical[row] = (planner_position, payload)
            else:
                raise ValueError("unknown path")
        decrypted = time.perf_counter_ns()
        sem_rows = np.asarray(sorted(semantic), dtype=np.int64)
        sem_scores = (np.stack([semantic[int(row)] for row in sem_rows]) @ embedding
                      if len(sem_rows) else np.asarray([]))
        # The new quality runner resolves equal native dot products by row ID.
        sem_order = np.lexsort((sem_rows, -sem_scores))[:self.config["semantic"]["output_depth"]]
        sem_rank = sem_rows[sem_order]
        semantic_done = time.perf_counter_ns()
        # Restore the FTS5 planner order BEFORE the declared cap of 300.
        scoring_cap = 1000 if lexical_mode == "direct1000" else self.config["lexical"]["exact_scoring_cap"]
        lex_rows = sorted(lexical, key=lambda row: lexical[row][0])[:scoring_cap]
        query_tokens = list(dict.fromkeys(tokenize(text)))
        pairs = [(self.private["token_ids"].get(term), self.private["document_frequencies"].get(keyed_term(term), 0))
                 for term in query_tokens]
        pairs = [(token, df) for token, df in pairs if token is not None and df > 0]
        n, average = self.cfg["documents"], self.private["average_document_length"]
        tf = np.zeros((len(lex_rows), len(pairs)), dtype=np.float64)
        lengths = np.zeros((len(lex_rows), 1), dtype=np.float64)
        for ordinal, row in enumerate(lex_rows):
            tokens = open_lexical_payload(lexical[row][1])
            counts = Counter(tokens)
            lengths[ordinal, 0] = len(tokens)
            for term_index, (token, _) in enumerate(pairs):
                tf[ordinal, term_index] = counts[token]
        dfs = np.asarray([df for _, df in pairs], dtype=np.float64)
        idf = np.log1p((n - dfs + 0.5) / (dfs + 0.5))[None, :]
        norm = 1.2 * (1.0 - 0.75 + 0.75 * lengths / average)
        scores = np.sum(idf * tf * 2.2 / np.maximum(tf + norm, 1e-12), axis=1)
        lex_array = np.asarray(lex_rows, dtype=np.int64)
        lex_order = np.lexsort((lex_array, -scores))
        lex_rank = lex_array[lex_order]
        lexical_done = time.perf_counter_ns()
        sem_padded = np.full((1, 100), -1, dtype=np.int32)
        lex_padded = np.full((1, scoring_cap), -1, dtype=np.int32)
        sem_padded[0, :len(sem_rank)] = sem_rank
        lex_padded[0, :len(lex_rank)] = lex_rank
        if semantic_enabled:
            if self.model is None:
                raise RuntimeError("frozen DuetRank model was not loaded")
            features = self.query_features([text], sem_padded[None], lex_padded[:, :100],
                {"terms": [trace["selected_terms"]], "estimated_postings": [trace["estimated_postings"]]})
            learned = self.learned_rankings(self.model, sem_padded, lex_padded, features)
            final = learned if backoff == 0 else self.mix_rankings(sem_padded, learned, backoff)
        else:
            final = lex_padded
        top20 = [int(row) for row in final[0, :20] if row >= 0]
        if len(top20) != len(set(top20)):
            raise ValueError("DuetRank produced duplicate Top20 IDs")
        finished = time.perf_counter_ns()
        return top20, [locators[row] for row in top20], {
            "payload_join_decrypt_dedup_ms": (decrypted - started) / 1e6,
            "semantic_exact_rerank_ms": (semantic_done - decrypted) / 1e6,
            "planner_restore_cap_exact_bm25_ms": (lexical_done - semantic_done) / 1e6,
            "duetrank_ms": (finished - lexical_done) / 1e6,
            "semantic_unique_candidates": len(semantic), "lexical_survivors": len(lexical),
            "lexical_exact_scored": len(lex_rows), "dummy_slots": dummy_count,
            "semantic_rank100": sem_padded[0].tolist(), "lexical_rank": lex_padded[0].tolist(),
        }


def measured_query(owner, sock, query, request_id, backoff, lexical_mode="ccadpe", semantic_enabled=True):
    start = time.perf_counter_ns()
    text = query["text"]
    embedding = owner.encode_text(text) if semantic_enabled else None
    encoded = time.perf_counter_ns()
    meta, blocks, trace = owner.request(text, query["id"], embedding, request_id, lexical_mode, semantic_enabled)
    response_meta, records, search_rpc = rpc(sock, meta, blocks)
    top20, locators, rerank = owner.rerank(text, embedding, response_meta, records, trace, backoff,
                                         lexical_mode, semantic_enabled)
    fetch_meta, encrypted_docs, fetch_rpc = rpc(sock,
        {"kind": "fetch", "request_id": request_id, "count": len(locators)}, [b"".join(locators)])
    decrypt_start = time.perf_counter_ns()
    documents = [owner.owner.open("document", data, locator)
                 for locator, data in zip(locators, encrypted_docs)]
    if len(documents) != len(top20) or fetch_meta["count"] != len(top20):
        raise ValueError("Fetch cardinality mismatch")
    # Actual full document bytes are decrypted, parsed and verified.
    parsed_docs = [json.loads(document) for document in documents]
    for row, document in zip(top20, parsed_docs):
        if str(document["_id"]) != owner.private["document_ids"][str(row)]:
            raise ValueError("Fetch returned the wrong plaintext document")
    end = time.perf_counter_ns()
    cached = np.asarray(query["cached_embedding"], dtype=np.float32)
    encoder_difference = None if embedding is None else {
        "max_abs": float(np.max(np.abs(embedding - cached))),
        "fresh_embedding_norm": float(np.linalg.norm(embedding)),
        "cached_embedding_norm": float(np.linalg.norm(cached)),
        "cosine": float(np.dot(embedding, cached) / max(np.linalg.norm(embedding) * np.linalg.norm(cached), 1e-12))}
    return {"request_id": request_id, "query_id": query["id"], "query_index": query["index"],
            "encoder_ms": (encoded - start) / 1e6, **trace, **rerank,
            "encoder_parameter_dtype": getattr(owner, "encoder_parameter_dtype", None),
            "encoder_output_dtype_before_float32_cast": getattr(owner, "last_encoder_output_dtype", None),
            "search_rpc": search_rpc, "search_server": response_meta["server"],
            "fetch_rpc": fetch_rpc, "fetch_server": fetch_meta["server"],
            "fetch_decrypt_parse_verify_ms": (end - decrypt_start) / 1e6,
            "end_to_end_ms": (end - start) / 1e6,
            "actual_application_upload_bytes": search_rpc["sent_application_bytes"] + fetch_rpc["sent_application_bytes"],
            "actual_application_download_bytes": search_rpc["received_application_bytes"] + fetch_rpc["received_application_bytes"],
            "fetched_documents": len(documents), "fetched_plaintext_bytes": sum(map(len, documents)),
            "encoder_vs_cached_diagnostic_outside_timer": encoder_difference,
            "top20_sha256": hashlib.sha256(json.dumps(top20).encode()).hexdigest(), "top20": top20}


def run(args):
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    private = json.loads((args.prepared / "client_private.json").read_text(encoding="utf-8"))
    if private["config_sha256"] != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError("prepared owner data does not match unified config")
    owner = QueryOwner(config, args.dataset, private, encoder_device=args.encoder_device)
    backoff = 0.0
    if not args.lexical_only:
        if not args.fusion_model or not args.fusion_report:
            raise ValueError("semantic+lexical mode requires its own frozen fusion model and report")
        owner.load_encoder()
        owner.load_ranker(args.fusion_model)
        fusion = json.loads(args.fusion_report.read_text(encoding="utf-8"))
        selection = fusion.get("selection", fusion.get("duetrank", {}).get("selection", {}))
        backoff = float(selection["semantic_backoff_weight"])
    ready = json.loads(args.server_ready.read_text(encoding="utf-8"))
    if ready["config_sha256"] != private["config_sha256"]:
        raise ValueError("server config mismatch")
    observations = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with socket.create_connection((ready["host"], ready["port"]), timeout=600) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for phase, passes in (("warmup", args.warmup_passes), ("measurement", args.measurement_passes)):
            for repeat in range(passes):
                for query in private["queries"]:
                    request_id = f"{phase}-{repeat}-{query['index']}"
                    observation = measured_query(owner, sock, query, request_id, backoff,
                                                 args.lexical_mode, not args.lexical_only)
                    observation.update(phase=phase, repeat=repeat)
                    observations.append(observation)
                    print(json.dumps({"request_id": request_id, "end_to_end_ms": observation["end_to_end_ms"]}), flush=True)
        if args.stop_server:
            rpc(sock, {"kind": "shutdown"}, [])
    selected = [item for item in observations if item["phase"] == "measurement"]
    values = np.asarray([item["end_to_end_ms"] for item in selected])
    args.output.with_suffix(".raw.json").write_text(json.dumps({
        "dataset": args.dataset, "config_sha256": private["config_sha256"],
        "lexical_mode": args.lexical_mode, "semantic_enabled": not args.lexical_only,
        "observations": observations}, indent=2, allow_nan=False), encoding="utf-8")
    # Diagnostic reads happen only AFTER all timed RPCs have completed. These
    # arrays never supply candidates, features or predictions to timed queries.
    reference_diagnostics = None
    if not args.lexical_only:
        fused_path = args.fusion_model.parent / "duetrank.rankings.npz"
        semantic_path = Path(config["output_root"]) / "results" / config["datasets"][args.dataset]["label"] / "semantic.rankings.npz"
        lexical_path = Path(fusion["lexical_input_path"])
        fused_reference = np.load(fused_path)["dual"][0]
        semantic_reference = np.load(semantic_path)["semantic"][0]
        lexical_reference = np.load(lexical_path)["lexical_dpe"][0]
        for item in observations:
            qi = item["query_index"]
            expected = [int(row) for row in fused_reference[qi, :20] if row >= 0]
            item["fresh_encoder_vs_cached_quality_postrun"] = {
                "top20_exact_match": item["top20"] == expected,
                "top20_set_overlap": len(set(item["top20"]) & set(expected)),
                "semantic_rank100_exact_match": item["semantic_rank100"] == semantic_reference[qi].tolist(),
                "lexical_rank_exact_match": item["lexical_rank"] == lexical_reference[qi, :len(item["lexical_rank"])].tolist(),
            }
        reference_diagnostics = {"scope": "16-query repeated diagnostic, not full-corpus utility evaluation; computed after timing",
            "fused_reference": str(fused_path), "semantic_reference": str(semantic_path),
            "lexical_reference": str(lexical_path),
            "measurement_top20_exact_matches": sum(item["fresh_encoder_vs_cached_quality_postrun"]["top20_exact_match"] for item in selected),
            "measurement_observations": len(selected)}
    report = {"dataset": args.dataset, "config_sha256": private["config_sha256"],
        "scope": ("Single persistent loopback TCP lexical-only client/server; text tokenization through Top20 Fetch document decryption; semantic encoder/fusion omitted"
                  if args.lexical_only else "Single persistent loopback TCP client/server; real text encoder through Top20 Fetch document decryption; no LLM generation or WAN"),
        "cache_policy": "warm process and prepared sample working set; OS/SQLite caches not flushed",
        "server_backend": ready["server_backend"], "encoder_device": args.encoder_device,
        "encoder_batch_size": 1 if not args.lexical_only else None,
        "encoder_normalize_embeddings": True if not args.lexical_only else None,
        "encoder_parameter_dtype": getattr(owner, "encoder_parameter_dtype", None),
        "encoder_output_dtype_before_float32_cast": getattr(owner, "last_encoder_output_dtype", None),
        "encoder_output_dtype_for_request_and_rerank": "float32" if not args.lexical_only else None,
        "semantic_rerank_similarity": "FP32 dot product with native stored payload vectors; no renormalization",
        "python_executable": sys.executable, "python_version": sys.version,
        "library_versions": {name: getattr(sys.modules.get(name), "__version__", None)
                             for name in ("numpy", "torch", "sentence_transformers", "transformers", "sklearn", "cryptography")},
        "warmup_passes": args.warmup_passes, "measurement_passes": args.measurement_passes,
        "sample_queries": len(private["queries"]), "query_view_seed_policy": private["query_view_seed_policy"],
        "lexical_mode": args.lexical_mode, "semantic_enabled": not args.lexical_only,
        "fusion_model": str(args.fusion_model) if args.fusion_model else None,
        "fusion_model_sha256": hashlib.sha256(args.fusion_model.read_bytes()).hexdigest() if args.fusion_model else None,
        "fusion_backoff": backoff, "semantic_payload_dtype": "float32",
        "noise": config["noise"], "client_pid": os.getpid(), "server_pid": ready["pid"],
        "record_store_sha256": private["record_store_sha256"],
        "server_index": ready["server_index"], "server_index_sha256": ready["server_index_sha256"],
        "lexical_coordinate_rows": ready["lexical_coordinate_rows"],
        "lexical_coordinate_scope": ready["lexical_coordinate_scope"],
        "encoder_path": private["encoder_path"],
        "protocol_source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in Path(__file__).resolve().parent.glob("*.py")},
        "timing_note": "end_to_end_ms is one client wall-clock interval; stage values overlap RPC/server spans and MUST NOT be added together",
        "byte_note": "actual serialized application bytes including framing; excludes TCP/IP/link-layer overhead",
        "fresh_encoder_vs_cached_quality_postrun": reference_diagnostics,
        "end_to_end_ms": {"mean": float(values.mean()), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))},
        "observations": observations}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("nq", "hotpotqa", "msmarco"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--server-ready", type=Path, required=True)
    parser.add_argument("--encoder-device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--fusion-model", type=Path)
    parser.add_argument("--fusion-report", type=Path)
    parser.add_argument("--lexical-mode", choices=("ccadpe", "direct300", "direct1000"), default="ccadpe")
    parser.add_argument("--lexical-only", action="store_true")
    parser.add_argument("--warmup-passes", type=int, default=2)
    parser.add_argument("--measurement-passes", type=int, default=3)
    parser.add_argument("--stop-server", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
