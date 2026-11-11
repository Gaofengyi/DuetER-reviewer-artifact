"""Serial non-timing validation of primary Duet and matched direct controls."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import traceback

from runtime import initialize
initialize(gpu_dependencies=True)
import validate


def main(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    protocol = Path(__file__).resolve().parent
    if args.output_dir.exists():
        raise FileExistsError("use a new validation directory to preserve earlier reports")
    args.output_dir.mkdir(parents=True)
    datasets = list(config["datasets"]) if args.dataset == "all" else [args.dataset]
    modes = ["ccadpe", "direct300", "direct1000"] if args.mode == "all" else [args.mode]
    reports = []
    for dataset in datasets:
        quality = root / "results" / config["datasets"][dataset]["label"]
        for mode in modes:
            model_dir = quality if mode == "ccadpe" else root / "direct_bm25" / dataset / f"fusion_{mode[6:]}"
            output = args.output_dir / f"{dataset}_{mode}.json"
            namespace = argparse.Namespace(config=args.config, dataset=dataset,
                prepared=protocol / "prepared" / f"{dataset}_cpu_v1",
                server_index=protocol / "server_indexes" / f"{dataset}.sqlite3",
                fusion_model=model_dir / "duetrank.joblib", fusion_report=model_dir / "fusion.json",
                reference_dir=quality,
                reference_lexical=None if mode == "ccadpe" else root / "direct_bm25" / dataset / f"direct_{mode[6:]}.rankings.npz",
                reference_fused=model_dir / "duetrank.rankings.npz", lexical_mode=mode, output=output,
                allow_local_backend_differences=args.allow_local_backend_differences)
            print(json.dumps({"validation_start": dataset, "mode": mode}), flush=True)
            try:
                validate.run(namespace)
                checked = json.loads(output.read_text(encoding="utf-8"))
                reports.append({"dataset": dataset, "mode": mode, "passed": True,
                    "all_intermediates_exact_match": checked["all_queries_match"],
                    "all_functional_checks_match": checked["all_functional_checks_match"], "report": str(output)})
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                reports.append({"dataset": dataset, "mode": mode, "passed": False,
                                "error": f"{type(exc).__name__}: {exc}", "report": str(output)})
                traceback.print_exc()
            gc.collect()
            (args.output_dir / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    if not all(item["passed"] for item in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("all", "nq", "hotpotqa", "msmarco"), default="all")
    parser.add_argument("--mode", choices=("all", "ccadpe", "direct300", "direct1000"), default="all")
    parser.add_argument("--allow-local-backend-differences", action="store_true")
    main(parser.parse_args())
