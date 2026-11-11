"""Release self-check: verifies that the compact paper artifacts are present,
parse as JSON, and contain the headline values reported in the paper.

Runs without datasets, models, or GPU. Exit code 0 means the release package
is internally consistent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results" / "paper"

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok " if name and False else ""
    if cond:
        print(f"[PASS] {name}" + (f" ({detail})" if detail else ""))
    else:
        print(f"[FAIL] {name}" + (f" ({detail})" if detail else ""))
        FAILURES.append(name)


def load(rel: str) -> object:
    path = R / rel
    check(f"exists: results/paper/{rel}", path.exists())
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        check(f"parses: results/paper/{rel}", False, str(exc))
        return None


def approx(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    # Table II / VI: unified quality comparison
    q = load("main/quality_comparison.json")
    if q:
        nq_direct300 = q["datasets"]["nq"]["methods"]["direct300"]["duetrank"]["heldout_ndcg10"]
        check("Table VI direct-300 fused nDCG@10 NQ ~= 0.5221", approx(nq_direct300, 0.5221), f"{nq_direct300:.5f}")
        hot_direct300 = q["datasets"]["hotpotqa"]["methods"]["direct300"]["duetrank"]["heldout_ndcg10"]
        check("Table VI direct-300 fused nDCG@10 HotpotQA ~= 0.6859", approx(hot_direct300, 0.6859), f"{hot_direct300:.5f}")

    # Table IV(b): probe-count sweep
    nprobe = load("ablations/nprobe.json")
    if nprobe:
        row128 = [r for r in nprobe["datasets"]["nq"]["rows"] if r["nprobe"] == 128][0]
        check("Table IV(b) NQ 128 probes nDCG@10 ~= 0.51833", approx(row128["nDCG@10"], 0.51833), f"{row128['nDCG@10']:.5f}")
        check("Table IV(b) NQ 128-probe scan ~= 13.19%", approx(row128["scan_percent"], 13.194, 0.01), f"{row128['scan_percent']:.2f}%")

    # Table IV(a): semantic local-depth sweep recorded in per-dataset results
    depth = load("ablations/semantic_depth_sweep/nq_results.json")
    if depth:
        sweep = {r["top_per_cell"]: r["metrics"] for r in depth["rq1_utility"]["semantic_sweep"]}
        ok = all(t in sweep for t in (8, 16, 32)) and approx(sweep[8]["nDCG@10"], 0.51723) and approx(sweep[32]["Recall@100"], 0.92539)
        check("Table IV(a) NQ depth sweep matches paper points (8/16/32)", ok,
              ", ".join(f"{t}:{sweep[t]['nDCG@10']:.5f}" for t in (8, 16, 32) if t in sweep))

    # Table V: calibration robustness
    cal = load("ablations/table5_duetrank_calibration_robustness.json")
    if cal:
        nq = [d for d in cal["datasets"] if d["dataset"] == "nq"][0]
        check("Table V NQ primary DuetRank outer nDCG@10 ~= 0.51910",
              approx(nq["primary_duetrank_outer_ndcg_at_10"], 0.51910),
              f"{nq['primary_duetrank_outer_ndcg_at_10']:.5f}")

    # Table VII: matched-budget components
    comp = load("components/table7_paired_results.json")
    if comp:
        full = comp["datasets"]["nq"]["metrics"]["full"]["ndcg10"]["mean"]
        check("Table VII full variant NQ nDCG@10 ~= 0.5185", approx(full, 0.5185), f"{full:.5f}")

    # Figure 3: protocol timing
    timing = load("efficiency/protocol_timing_summary.json")
    if timing:
        check("Figure 3 protocol timing status complete", timing.get("status") == "complete")
        check("Figure 3 protocol timing has 9 formal runs",
              timing.get("invariant_checks", {}).get("formal_runs") == 9)

    # Figure 2 data
    fig2 = load("main/figure2_path_utility_data.json")
    check("Figure 2 data parses", fig2 is not None)

    # Figure 4 security artifacts
    for rel in ["security/cross_path_anchor_fingerprint.json",
                "security/cross_path_cooccurrence.json",
                "security/cross_path_posting_access.json",
                "security/known_pair_recovery.json"]:
        load(rel)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed.")
        return 1
    print("All release checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
