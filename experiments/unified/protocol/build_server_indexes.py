"""Copy historical FTS5 indexes read-only; remove ID metadata only in isolated copies.

The original files are never opened writable. No benchmark query or result changes.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import csv
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import sqlite3
import time

PROTOCOL = Path(__file__).resolve().parent
ROOT = PROTOCOL / "server_indexes"
CONFIG = PROTOCOL.parent / "config.json"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
LEXICAL_KEY = b"DuetDPE-keyed-vocabulary-tail-v2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def readonly(path: Path):
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def schema(connection):
    return connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name").fetchall()


def counts(connection):
    # FTS5 contentless indexes keep one docsize entry per document.
    count, smallest, largest = connection.execute("SELECT count(*),min(id),max(id) FROM postings_docsize").fetchone()
    return {"documents": count, "minimum_rowid": smallest, "maximum_rowid": largest}


def sample_queries(config, dataset):
    dc = config["datasets"][dataset]
    data = Path(config["original_root"]) / "data" / dataset
    with (data / "qrels" / f"{dc['split']}.tsv").open(encoding="utf-8", newline="") as f:
        ids = {str(row["query-id"]) for row in csv.DictReader(f, delimiter="\t")}
    queries = []
    with (data / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            qid = str(q["_id"])
            if qid in ids and hashlib.sha256(("outer:" + qid).encode()).digest()[0] & 1:
                queries.append((qid, q["text"]))
    assert len(queries) == dc["outer_holdout_queries"]
    return [queries[round(i * (len(queries) - 1) / 15)] for i in range(16)]


def planner_sample(connection, text, dc):
    labels = list(dict.fromkeys(hmac.new(LEXICAL_KEY, t.encode(), hashlib.sha256).hexdigest()[:32]
                               for t in TOKEN_RE.findall(text.lower())))
    freq = dict(connection.execute(
        "SELECT term,doc FROM vocab WHERE term IN (" + ",".join("?" for _ in labels) + ")", labels).fetchall()) if labels else {}
    ranked = sorted(freq, key=lambda term: (-math.log1p(dc["documents"] / max(freq[term], 1)) / freq[term] ** .5, freq[term], term))
    selected, spent = [], 0
    for term in ranked:
        if not selected or spent + freq[term] <= dc["posting_budget"]:
            selected.append(term)
            spent += freq[term]
    expression = " OR ".join(sorted(selected))
    rows = connection.execute("SELECT rowid FROM postings WHERE postings MATCH ? ORDER BY bm25(postings) LIMIT 1000", (expression,)).fetchall() if expression else []
    return {"labels": sorted(selected), "estimated_postings": spent, "rowids": [r[0] for r in rows]}


def build(dataset):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    dc = cfg["datasets"][dataset]
    source = (Path(cfg["original_root"]) / dc["fts_database"]).resolve(strict=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    isolated = ROOT.resolve(strict=True)
    target = (isolated / f"{dataset}.sqlite3").resolve()
    temporary = (isolated / f"{dataset}.building.sqlite3").resolve()
    # This checked boundary precedes every DROP/VACUUM and publication operation.
    assert target.parent == isolated and temporary.parent == isolated
    assert source not in (target, temporary) and isolated not in source.parents
    if target.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output: {target} or {temporary}")
    started = time.perf_counter()
    state_path = isolated / f"{dataset}.progress.json"
    report = {"dataset": dataset, "source": str(source), "target": str(target),
              "config_sha256": sha256(CONFIG), "script_sha256": sha256(Path(__file__)),
              "sqlite_version": sqlite3.sqlite_version, "status": "in_progress"}

    def state(stage):
        atomic_json(state_path, {**report, "stage": stage, "elapsed_seconds": time.perf_counter() - started})
        print(f"[{dataset}] {stage}", flush=True)

    state("read_only_source_hash_and_samples")
    source_stat = source.stat()
    report["source_bytes"] = source_stat.st_size
    report["source_sha256_before"] = sha256(source)
    src = readonly(source)
    source_schema = schema(src)
    assert any(row[1] == "docmap" and row[0] == "table" for row in source_schema)
    source_counts = counts(src)
    assert source_counts["documents"] == dc["documents"]
    queries = sample_queries(cfg, dataset)
    reference = [(qid, planner_sample(src, text, dc)) for qid, text in queries]
    report["source_integrity_check"] = [row[0] for row in src.execute("PRAGMA integrity_check").fetchall()]
    assert report["source_integrity_check"] == ["ok"]
    state("sqlite_backup_to_checked_isolated_path")
    dst = sqlite3.connect(temporary)
    last = [-1]

    def progress(status, remaining, total):
        percent = int(100 * (total - remaining) / max(total, 1))
        if percent // 20 != last[0]:
            last[0] = percent // 20
            print(f"[{dataset}] backup {percent}%", flush=True)

    src.backup(dst, pages=8192, progress=progress)
    dst.close()
    src.close()
    state("drop_docmap_in_new_copy_and_vacuum")
    assert temporary.resolve().parent == isolated and temporary.resolve() != source
    with closing(sqlite3.connect(temporary)) as dst:
        copied_schema = schema(dst)
        assert copied_schema == source_schema
        dst.execute("PRAGMA secure_delete=ON")
        dst.execute("DROP TABLE docmap")  # Its associated autoindex is removed by SQLite.
        dst.commit()
        dst.execute("VACUUM")
        dst.commit()
    state("verify_copy_integrity_schema_counts_and_planner")
    with closing(readonly(temporary)) as dst:
        target_schema = schema(dst)
        expected_schema = [row for row in source_schema if row[2] != "docmap"]
        assert target_schema == expected_schema, "Only docmap and its own indexes may be removed"
        assert not dst.execute("SELECT 1 FROM sqlite_schema WHERE name='docmap' OR tbl_name='docmap'").fetchall()
        report["target_integrity_check"] = [row[0] for row in dst.execute("PRAGMA integrity_check").fetchall()]
        assert report["target_integrity_check"] == ["ok"]
        report["target_freelist_pages"] = dst.execute("PRAGMA freelist_count").fetchone()[0]
        assert report["target_freelist_pages"] == 0
        target_counts = counts(dst)
        assert target_counts == source_counts
        samples = []
        for (qid, text), (reference_id, before) in zip(queries, reference):
            after = planner_sample(dst, text, dc)
            assert qid == reference_id and before == after, f"Planner sequence mismatch for {qid}"
            samples.append({"query_id": qid, "selected_terms": len(before["labels"]),
                            "estimated_postings": before["estimated_postings"], "candidate_count": len(before["rowids"]),
                            "rowid_sequence_sha256": digest(before["rowids"]), "exact_sequence_equal": True})
    state("recheck_original_hash_then_publish_copy")
    report["source_sha256_after"] = sha256(source)
    assert report["source_sha256_before"] == report["source_sha256_after"]
    assert source.stat().st_size == source_stat.st_size and source.stat().st_mtime_ns == source_stat.st_mtime_ns
    report.update({"source_unchanged": True, "target_sha256": sha256(temporary),
                   "target_bytes": temporary.stat().st_size, "source_schema": source_schema,
                   "target_schema": target_schema, "only_docmap_and_its_indexes_removed": True,
                   "docmap_absent": True, "source_fts_counts": source_counts, "target_fts_counts": target_counts,
                   "samples": samples, "sample_policy": "16 evenly spaced actual outer-holdout queries; same deterministic selected-label OR query and native FTS5 BM25 order, cap1000",
                   "position_metadata_retained": True,
                   "scope": "Removes the docmap original-corpus-ID mapping and its freelist remnants only; retains FTS5 term positions, frequencies, lengths, rowids, shadows and vocab tables. Does not claim removal of all inference leakage or a production-isolated filesystem.",
                   "elapsed_seconds": time.perf_counter() - started, "status": "passed"})
    if target.exists():
        raise FileExistsError(f"Refusing to replace newly appeared output: {target}")
    temporary.rename(target)
    atomic_json(isolated / f"{dataset}.manifest.json", report)
    atomic_json(state_path, {"dataset": dataset, "status": "passed", "stage": "published", "target": str(target)})
    print(json.dumps({"dataset": dataset, "status": "passed", "source_bytes": report["source_bytes"],
                      "target_bytes": report["target_bytes"], "target_sha256": report["target_sha256"],
                      "queries_checked": len(samples), "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nq", "hotpotqa", "msmarco"], required=True)
    build(parser.parse_args().dataset)
