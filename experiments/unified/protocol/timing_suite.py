"""Nine sequential formal runs. Start only after the shared-resource handoff."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import run_loopback


def main(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    protocol = Path(__file__).resolve().parent
    if args.output_dir.exists():
        raise FileExistsError("a formal timing suite must use a new output directory")
    args.output_dir.mkdir(parents=True)
    datasets = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    modes = ["ccadpe", "direct300", "direct1000"] if args.mode == "all" else [args.mode]
    validated = json.loads((protocol / "validation/final_v2/summary.json").read_text(encoding="utf-8"))
    for dataset in datasets:
        for mode in modes:
            matching = [row for row in validated if row["dataset"] == dataset and row["mode"] == mode]
            if len(matching) != 1 or not matching[0]["passed"] or not matching[0]["all_functional_checks_match"]:
                raise RuntimeError(f"missing successful functional validation for {dataset}/{mode}")
    summary = {"started_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "16 evenly spaced queries; 2 complete warmup passes and 3 measured passes; CPU single-query encoder; CPU server; sequential loopback client/server runs", "runs": []}
    for dataset in datasets:
        quality = root / "results" / config["datasets"][dataset]["label"]
        for mode in modes:
            model = quality if mode == "ccadpe" else root / "direct_bm25" / dataset / f"fusion_{mode[6:]}"
            destination = args.output_dir / f"{dataset}_{mode}"
            print(json.dumps({"formal_timing_start": dataset, "mode": mode}), flush=True)
            namespace = argparse.Namespace(config=args.config, dataset=dataset,
                prepared=protocol / "prepared" / f"{dataset}_cpu_v1",
                server_index=protocol / "server_indexes" / f"{dataset}.sqlite3",
                output_dir=destination, encoder_device="cpu", lexical_mode=mode, lexical_only=False,
                fusion_model=model / "duetrank.joblib", fusion_report=model / "fusion.json",
                warmup_passes=2, measurement_passes=3)
            run_loopback.main(namespace)
            report = json.loads((destination / "results.json").read_text(encoding="utf-8"))
            measurements = [row for row in report["observations"] if row["phase"] == "measurement"]
            if len(measurements) != 48 or report["sample_queries"] != 16:
                raise AssertionError("formal timing sample/pass count differs from the registered design")
            summary["runs"].append({"dataset": dataset, "mode": mode, "report": str(destination / "results.json"),
                "end_to_end_ms": report["end_to_end_ms"],
                "encoder_parameter_dtype": report["encoder_parameter_dtype"],
                "mean_application_upload_bytes": sum(row["actual_application_upload_bytes"] for row in measurements) / 48,
                "mean_application_download_bytes": sum(row["actual_application_download_bytes"] for row in measurements) / 48,
                "fresh_encoder_postrun": report["fresh_encoder_vs_cached_quality_postrun"]})
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps({"formal_timing_complete": dataset, "mode": mode,
                              "end_to_end_ms": report["end_to_end_ms"]}), flush=True)
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset", choices=("all", "nq", "hotpotqa", "msmarco"), default="all")
    p.add_argument("--mode", choices=("all", "ccadpe", "direct300", "direct1000"), default="all")
    main(p.parse_args())
