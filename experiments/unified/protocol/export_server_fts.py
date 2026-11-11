"""Creation-only export of FTS5 live shadow rows; never copy document-ID metadata."""
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
import sys
import time

BASE = Path(__file__).resolve().parent
OUT = BASE / "server_indexes"
CONFIG = BASE.parent / "config.json"
TABLES = ("postings_data", "postings_idx", "postings_docsize", "postings_config")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)

def ro(path):
    c = sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True)
    c.execute("PRAGMA query_only=ON")
    return c

def schema(c):
    return c.execute("SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name").fetchall()

def counts(c):
    n, minimum, maximum = c.execute("SELECT count(*),min(id),max(id) FROM postings_docsize").fetchone()
    return {"documents": n, "minimum_rowid": minimum, "maximum_rowid": maximum}

def queries(cfg, name):
    dc = cfg["datasets"][name]
    data = Path(cfg["original_root"]) / "data" / name
    with (data / "qrels" / f"{dc['split']}.tsv").open(encoding="utf-8", newline="") as f:
        ids = {str(x["query-id"]) for x in csv.DictReader(f, delimiter="\t")}
    rows = []
    with (data / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            qid = str(q["_id"])
            if qid in ids and hashlib.sha256(("outer:" + qid).encode()).digest()[0] & 1:
                rows.append((qid, q["text"]))
    assert len(rows) == dc["outer_holdout_queries"]
    return [rows[round(i * (len(rows) - 1) / 15)] for i in range(16)]

def planner(c, text, dc):
    labels = list(dict.fromkeys(hmac.new(b"DuetDPE-keyed-vocabulary-tail-v2", t.encode(), hashlib.sha256).hexdigest()[:32]
                               for t in re.findall(r"[A-Za-z0-9]+", text.lower())))
    frequencies = dict(c.execute("SELECT term,doc FROM vocab WHERE term IN (" + ",".join("?" for _ in labels) + ")", labels).fetchall()) if labels else {}
    ranked = sorted(frequencies, key=lambda t: (-math.log1p(dc["documents"] / max(frequencies[t], 1)) / frequencies[t] ** .5, frequencies[t], t))
    selected, spent = [], 0
    for term in ranked:
        if not selected or spent + frequencies[term] <= dc["posting_budget"]:
            selected.append(term)
            spent += frequencies[term]
    selected.sort()
    rows = c.execute("SELECT rowid FROM postings WHERE postings MATCH ? ORDER BY bm25(postings) LIMIT 1000", (" OR ".join(selected),)).fetchall() if selected else []
    return {"labels": selected, "estimated_postings": spent, "rowids": [r[0] for r in rows]}

def run(name, preflight=False):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    dc = cfg["datasets"][name]
    source = (Path(cfg["original_root"]) / dc["fts_database"]).resolve(strict=True)
    if preflight:
        with closing(ro(source)) as c:
            print(json.dumps({"dataset": name, "python": sys.version, "sqlite": sqlite3.sqlite_version,
                              "fts5": c.execute("SELECT sqlite_compileoption_used(?)", ("ENABLE_FTS5",)).fetchone()[0],
                              "vocab_read_ok": c.execute("SELECT doc FROM vocab LIMIT 1").fetchone() is not None,
                              "source_schema": schema(c)}), flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    root = OUT.resolve(strict=True)
    target, temporary = root / f"{name}.sqlite3", root / f"{name}.exporting.sqlite3"
    assert target.resolve().parent == root and temporary.resolve().parent == root and root not in source.parents
    if target.exists() or temporary.exists():
        raise FileExistsError("Refusing to overwrite an existing final or temporary index")
    started = time.perf_counter()
    report = {"dataset": name, "source": str(source), "target": str(target), "python": sys.version,
              "sqlite_version": sqlite3.sqlite_version, "script_sha256": sha(__file__), "config_sha256": sha(CONFIG),
              "construction": "New SQLite FTS5 database with copied live FTS shadow rows only; docmap is never copied; no table deletion or VACUUM", "status": "in_progress"}

    def stage(label):
        save(root / f"{name}.progress.json", {**report, "stage": label, "elapsed_seconds": time.perf_counter() - started})
        print(f"[{name}] {label}", flush=True)

    stage("read_only_source_hash_integrity_and_samples")
    original_stat = source.stat()
    report["source_bytes"] = original_stat.st_size
    report["source_sha256_before"] = sha(source)
    sample_queries = queries(cfg, name)
    with closing(ro(source)) as src:
        source_schema = schema(src)
        source_counts = counts(src)
        assert source_counts["documents"] == dc["documents"]
        report["source_integrity_check"] = [r[0] for r in src.execute("PRAGMA integrity_check")]
        assert report["source_integrity_check"] == ["ok"]
        references = [planner(src, text, dc) for _, text in sample_queries]
        # FTS5 initializes these two data records; replacing them leaves no extra empty-index rows.
        assert set(x[0] for x in src.execute("SELECT id FROM postings_data WHERE id IN (1,10)")) == {1, 10}
    stage("create_empty_fts_and_copy_live_shadows_from_read_only_attachment")
    with closing(sqlite3.connect(temporary, uri=True)) as dst:
        postings_sql = next(r[3] for r in source_schema if r[1] == "postings")
        dst.execute(postings_sql)
        dst.commit()
        dst.execute("ATTACH DATABASE ? AS original", (source.as_uri() + "?mode=ro",))
        dst.execute("BEGIN")
        for table in TABLES:
            print(f"[{name}] copying {table}", flush=True)
            dst.execute(f'INSERT OR REPLACE INTO main."{table}" SELECT * FROM original."{table}"')
        dst.commit()
        dst.execute("DETACH DATABASE original")
        for kind, table, _, sql in source_schema:
            if kind == "table" and sql and "USING fts5vocab(" in sql:
                dst.execute(sql)
        dst.commit()
    stage("verify_reopened_copy_integrity_schema_counts_and_planner")
    with closing(ro(temporary)) as dst:
        target_schema = schema(dst)
        assert target_schema == [r for r in source_schema if r[2] != "docmap"]
        assert not dst.execute("SELECT 1 FROM sqlite_schema WHERE name='docmap' OR tbl_name='docmap'").fetchall()
        report["target_integrity_check"] = [r[0] for r in dst.execute("PRAGMA integrity_check")]
        assert report["target_integrity_check"] == ["ok"]
        target_counts = counts(dst)
        assert target_counts == source_counts
        report["target_freelist_pages"] = dst.execute("PRAGMA freelist_count").fetchone()[0]
        checked = []
        for (qid, text), reference in zip(sample_queries, references):
            actual = planner(dst, text, dc)
            assert actual == reference, f"Planner sequence mismatch: {qid}"
            checked.append({"query_id": qid, "selected_terms": len(actual["labels"]),
                            "estimated_postings": actual["estimated_postings"], "candidate_count": len(actual["rowids"]),
                            "rowid_sequence_sha256": digest(actual["rowids"]), "exact_sequence_equal": True})
    stage("verify_original_unchanged_then_publish_copy")
    report["source_sha256_after"] = sha(source)
    assert report["source_sha256_before"] == report["source_sha256_after"]
    assert source.stat().st_size == original_stat.st_size and source.stat().st_mtime_ns == original_stat.st_mtime_ns
    report.update({"source_unchanged": True, "target_bytes": temporary.stat().st_size, "target_sha256": sha(temporary),
                   "source_schema": source_schema, "target_schema": target_schema, "docmap_absent": True,
                   "docmap_never_copied": True, "source_fts_counts": source_counts, "target_fts_counts": target_counts,
                   "samples": checked, "sample_policy": "16 evenly spaced actual outer-holdout queries; exact selected-label and native FTS5 BM25 order equality at cap1000",
                   "position_metadata_retained": True, "scope": "Exports FTS5 live data, positions, frequencies, lengths, rowids and vocab tables. Removes no declared FTS leakage; excludes original corpus-ID docmap by construction. Does not establish production filesystem isolation.",
                   "status": "passed", "elapsed_seconds": time.perf_counter() - started})
    if target.exists():
        raise FileExistsError("Final path appeared during export; refusing to replace")
    temporary.rename(target)
    save(root / f"{name}.manifest.json", report)
    save(root / f"{name}.progress.json", {"dataset": name, "status": "passed", "stage": "published", "target": str(target)})
    print(json.dumps({"dataset": name, "status": "passed", "target_bytes": report["target_bytes"],
                      "target_sha256": report["target_sha256"], "queries_checked": len(checked), "seconds": report["elapsed_seconds"]}), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nq", "hotpotqa", "msmarco"])
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    run(args.dataset, args.preflight)
