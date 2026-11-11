"""Load existing dependency directories without modifying the old project."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys


def initialize(gpu_dependencies=False):
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    path = Path(__file__).resolve().parent.parent / "config.json"
    if "--config" in sys.argv:
        path = Path(sys.argv[sys.argv.index("--config") + 1])
    config = json.loads(path.read_text(encoding="utf-8"))
    root = Path(config["original_root"])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(config["runtime"]["blas_threads"])
    dependencies = [root / ".deps"]
    if gpu_dependencies:
        dependencies.insert(0, root / ".gpu_deps")
    dependencies.insert(0, Path(__file__).resolve().parent / ".deps")
    sys.path[:0] = [str(item) for item in dependencies if item.exists()]
    return config
