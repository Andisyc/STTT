"""Reuse the server CPU/cgroup envelope for detached, read-only diagnostics."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from launch_latent_pretraining import ROOT, choose_backend, inspect_resources, launch


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="后台执行控制诊断或复用扫描重跑反馈，日志写入独立文件。")
    parser.add_argument("--run-dir", type=Path, required=True, help="summary.json 指向阶段2 WM 与 A/FiLM 存档的训练目录")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("all", "physical", "analyze", "feedback"), default="all")
    parser.add_argument("--data-dir", type=Path, help="analyze/feedback 模式使用的既有诊断目录")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="只打印资源预算和实际命令，不启动进程")
    options = parser.parse_args(argv)
    run = options.run_dir.resolve()
    if not (run / "summary.json").is_file():
        raise FileNotFoundError(run / "summary.json")
    if options.mode not in {"physical", "feedback"}:
        summary = json.loads((run / "summary.json").read_text())
        for field in ("world_model_checkpoint", "actor_checkpoint"):
            checkpoint = Path(summary[field])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
    if options.mode in {"analyze", "feedback"} and (options.data_dir is None or not options.data_dir.is_dir()):
        raise ValueError("analyze/feedback 模式需要 --data-dir 指向既有诊断数据")
    output = (options.output_dir or ROOT / f"outputs/latent_control_diagnostics_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    uv, ionice = shutil.which("uv"), shutil.which("ionice")
    if uv is None or ionice is None:
        raise RuntimeError("需要已安装的 uv 和 ionice；不会安装工具")
    command = [uv, "run", "--no-sync", "scripts/diagnose_latent_control.py", f"mode={options.mode}",
        f"run_dir={json.dumps(str(run))}", f"output_dir={json.dumps(str(output))}", f"device={options.device}",
        *([] if options.data_dir is None else [f"data_dir={json.dumps(str(options.data_dir.resolve()))}"])]
    cgroup, resources = inspect_resources()
    environment = dict(os.environ)
    # One deterministic physical branch at a time; tiny batch-one networks do not benefit from a large BLAS pool.
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    backend = choose_backend(cgroup, resources, environment)
    budget = asdict(resources)
    budget["constrained_groups"] = [str(path) for path in resources.constrained_groups]
    print(json.dumps({"resources": budget, "backend": backend, "output": str(output), "log": str(output) + ".log",
        "command": command, "dry_run": options.dry_run}, ensure_ascii=False), flush=True)
    if options.dry_run:
        return
    print(launch(command, output, cgroup, resources, backend, environment, uv, ionice), flush=True)
    print(f"查看日志：tail -f {str(output) + '.log'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f"诊断启动失败：{error}") from error
