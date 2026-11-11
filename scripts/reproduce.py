"""DuetER reproduction driver.

Prints the exact commands for each release profile. Commands are only executed
when ``--execute`` is supplied, so the workload is visible before a potentially
long or storage-intensive run. Full-corpus profiles assume the datasets,
encoder snapshot, and prepared caches described in REPRODUCIBILITY.md.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "experiments" / "core"
UNIFIED = ROOT / "experiments" / "unified"

PROG = "python"  # printed; replaced by the running interpreter under --execute

PROFILES: dict[str, list[tuple[str, list[str]]]] = {
    "quick": [
        ("release self-check", [PROG, "scripts/check_release.py"]),
        ("core unit tests", [PROG, "-m", "unittest", "discover", "-s", "tests", "-v"]),
    ],
    "full": [
        ("download and verify BEIR archives",
         [PROG, str(CORE / "download_beir.py"), "msmarco", "nq", "hotpotqa"]),
        ("unified quality prerequisites check",
         [PROG, str(UNIFIED / "check_quality_prerequisites.py")]),
        ("semantic stage (all datasets)",
         [PROG, str(UNIFIED / "run_quality_stage.py"), "--stage", "semantic"]),
        ("lexical stage", [PROG, str(UNIFIED / "run_quality_stage.py"), "--stage", "lexical"]),
        ("exact BM25 stage", [PROG, str(UNIFIED / "run_quality_stage.py"), "--stage", "bm25"]),
        ("fusion stage", [PROG, str(UNIFIED / "run_quality_stage.py"), "--stage", "fusion"]),
        ("direct-BM25 controls", [PROG, str(UNIFIED / "run_direct_bm25.py")]),
        ("direct-fusion controls", [PROG, str(UNIFIED / "run_direct_fusion.py")]),
        ("quality summary and verification",
         [PROG, str(UNIFIED / "summarize_quality.py")]),
    ],
    "efficiency": [
        ("loopback protocol timing (16 queries, 2 warmup + 3 measured passes)",
         [PROG, str(UNIFIED / "protocol" / "timing_suite.py")]),
        ("protocol validation",
         [PROG, str(UNIFIED / "protocol" / "validate_suite.py")]),
    ],
    "security": [
        ("cross-view linkage and geometry attacks",
         [PROG, str(CORE / "run_security_attacks.py")]),
        ("compartment security suite",
         [PROG, str(CORE / "run_compartment_security.py")]),
        ("known-pair recovery stress test",
         [PROG, str(CORE / "analyze_known_pair_recovery.py")]),
        ("lexical stitching attacks",
         [PROG, str(CORE / "attack_lexical_stitching_d8192_rc32.py")]),
        ("semantic parameter mitigation sweep",
         [PROG, str(CORE / "attack_semantic_parameter_mitigation.py")]),
    ],
    "ablations": [
        ("semantic probe-count sweep (Table IVb)",
         [PROG, str(CORE / "ablate_compartment_nprobe.py")]),
        ("larger lexical local-depth points (Table IV)",
         [PROG, str(CORE / "ablate_compartment_larger_lexical_depth.py")]),
        ("DuetRank calibration robustness (Table V)",
         [PROG, str(CORE / "experiment_duetrank_calibration_robustness.py")]),
        ("lexical dimension sweep (Table VI)",
         [PROG, str(CORE / "evaluate_lexical_d8192_rc32_b_sweep.py")]),
        ("direct-300 lexical-selection control (Table VI)",
         [PROG, str(UNIFIED / "run_direct_bm25.py")]),
    ],
    "components": [
        ("matched-budget component ablation (Table VII)",
         [PROG, str(ROOT / "experiments" / "components" / "run_components.py")]),
        ("component quality stage",
         [PROG, str(ROOT / "experiments" / "components" / "quality" / "run_quality.py")]),
        ("component validation",
         [PROG, str(ROOT / "experiments" / "components" / "validate_components.py")]),
    ],
}
PROFILES["all"] = [step for name in ("quick", "full", "efficiency", "security", "ablations", "components") for step in PROFILES[name]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--execute", action="store_true",
                        help="actually run the commands (default: print only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands without running (default)")
    args = parser.parse_args()

    steps = PROFILES[args.profile]
    for i, (label, cmd) in enumerate(steps, 1):
        print(f"# [{i}/{len(steps)}] {label}")
        print(" ".join(cmd))
        if args.execute:
            print(f"# executing: {label}", flush=True)
            run = [sys.executable if i == 0 and c == PROG else c for i, c in enumerate(cmd)]
            subprocess.run(run, cwd=ROOT, check=True)
        print()
    if not args.execute:
        print("Dry run only. Re-run with --execute after datasets, models, and")
        print("caches are prepared as described in REPRODUCIBILITY.md.")


if __name__ == "__main__":
    main()
