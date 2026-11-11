"""Read-only full logical equality check of all FTS5 shadow-table records."""
from contextlib import closing
from pathlib import Path
import json
import sqlite3
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE / "server_indexes"
definitions = {
    "postings_data": ("a.id=b.id", "a.block IS b.block"),
    "postings_idx": ("a.segid=b.segid AND a.term=b.term", "a.pgno IS b.pgno"),
    "postings_docsize": ("a.id=b.id", "a.sz IS b.sz"),
    "postings_config": ("a.k=b.k", "a.v IS b.v"),
}
report = {"status": "in_progress", "sqlite_version": sqlite3.sqlite_version, "datasets": {}}
started = time.perf_counter()
for dataset in ("nq", "hotpotqa", "msmarco"):
    manifest = json.loads((ROOT / f"{dataset}.manifest.json").read_text(encoding="utf-8"))
    target, source = Path(manifest["target"]), Path(manifest["source"])
    checks = {}
    with closing(sqlite3.connect(target.as_uri() + "?mode=ro", uri=True)) as c:
        c.execute("PRAGMA query_only=ON")
        c.execute("ATTACH DATABASE ? AS original", (source.as_uri() + "?mode=ro",))
        for table, (join, equals) in definitions.items():
            nsource = c.execute(f'SELECT count(*) FROM original."{table}"').fetchone()[0]
            ntarget = c.execute(f'SELECT count(*) FROM main."{table}"').fetchone()[0]
            nidentical = c.execute(f'SELECT count(*) FROM original."{table}" a JOIN main."{table}" b ON {join} WHERE {equals}').fetchone()[0]
            assert nsource == ntarget == nidentical, (dataset, table, nsource, ntarget, nidentical)
            checks[table] = {"source_rows": nsource, "target_rows": ntarget, "all_values_equal_rows": nidentical, "passed": True}
            print(f"{dataset} {table}: {nidentical} / {nsource} rows identical", flush=True)
    report["datasets"][dataset] = checks
report.update(status="passed", elapsed_seconds=time.perf_counter() - started,
              scope="Read-only primary-key joins compare every FTS5 shadow row and BLOB/value byte equality. Equal counts plus all-row equality establish identical live FTS index data; file layouts and docmap intentionally differ.")
(ROOT / "full_fts_row_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("FULL FTS ROW EQUALITY PASSED", flush=True)
