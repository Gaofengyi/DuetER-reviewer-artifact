"""Schedule ONE complete local client/server run; no detached helper survives.

Run only when the root coordinator has released the shared GPU/disk slot.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(args):
    here = Path(__file__).resolve().parent
    if args.output_dir.exists():
        raise FileExistsError("choose a new output directory to preserve all prior runs")
    args.output_dir.mkdir(parents=True)
    ready = args.output_dir / "server_ready.json"
    config = json.loads(args.config.read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(config["runtime"]["blas_threads"])
    common = ["--config", str(args.config), "--dataset", args.dataset, "--prepared", str(args.prepared)]
    server_command = [sys.executable, "-u", str(here / "server.py"), *common,
                      "--ready", str(ready), "--log", str(args.output_dir / "server_requests.jsonl")]
    if args.server_index:
        server_command += ["--server-index", str(args.server_index)]
    client_command = [sys.executable, "-u", str(here / "client.py"), *common,
        "--server-ready", str(ready), "--encoder-device", args.encoder_device,
        "--lexical-mode", args.lexical_mode,
        "--warmup-passes", str(args.warmup_passes), "--measurement-passes", str(args.measurement_passes),
        "--stop-server", "--output", str(args.output_dir / "results.json")]
    if args.lexical_only:
        client_command.append("--lexical-only")
    else:
        if not args.fusion_model or not args.fusion_report:
            raise ValueError("complete Duet run requires matching frozen fusion files")
        client_command += ["--fusion-model", str(args.fusion_model), "--fusion-report", str(args.fusion_report)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (args.output_dir / "server.log").open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(server_command, stdout=server_log, stderr=subprocess.STDOUT,
                                  env=env, creationflags=flags)
        try:
            deadline = time.monotonic() + 120
            while not ready.exists():
                if server.poll() is not None:
                    raise RuntimeError("server failed before ready; see server.log")
                if time.monotonic() > deadline:
                    raise TimeoutError("server did not become ready")
                time.sleep(.1)
            print(f"server ready pid={server.pid}", flush=True)
            with (args.output_dir / "client.log").open("w", encoding="utf-8") as client_log:
                result = subprocess.run(client_command, stdout=client_log, stderr=subprocess.STDOUT,
                                        env=env, creationflags=flags)
            if result.returncode:
                raise RuntimeError(f"client failed with exit {result.returncode}; see client.log")
            server.wait(timeout=30)
            print(str(args.output_dir / "results.json"), flush=True)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=("nq", "hotpotqa", "msmarco"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--server-index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--fusion-model", type=Path)
    parser.add_argument("--fusion-report", type=Path)
    parser.add_argument("--lexical-mode", choices=("ccadpe", "direct300", "direct1000"), default="ccadpe")
    parser.add_argument("--lexical-only", action="store_true")
    parser.add_argument("--warmup-passes", type=int, default=2)
    parser.add_argument("--measurement-passes", type=int, default=3)
    main(parser.parse_args())
