"""Launch the existing latent pretraining entrypoint with a Linux resource envelope.

Run with ``uv run --no-sync scripts/launch_latent_pretraining.py --help``.
This launcher neither stops existing jobs nor changes the training schedule.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

GIB = 1024**3
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Cgroup:
    mount_root: Path
    mount: Path
    current: Path

    def locate(self, name: str) -> Path:
        path = Path(name)
        if not path.is_absolute() or ".." in path.parts:
            raise RuntimeError("无法安全映射当前 cgroup 路径。")
        if path.is_relative_to(self.mount_root):
            result = self.mount / path.relative_to(self.mount_root)
        elif path == Path("/"):
            result = self.mount
        else:
            raise RuntimeError("cgroup 位于当前挂载命名空间之外。")
        if not result.is_dir():
            raise RuntimeError(f"cgroup 路径不可访问：{result}")
        return result

    def ancestors(self):
        if not self.current.is_relative_to(self.mount):
            raise RuntimeError("当前 cgroup 不在识别到的挂载点内。")
        node = self.current
        while True:
            yield node
            if node == self.mount:
                return
            node = node.parent


@dataclass(frozen=True)
class Resources:
    cpus: tuple[int, ...]
    reserved_cpus: tuple[int, ...]
    reserved_cores: int
    threads: int
    memory_bytes: int
    available_memory_bytes: int
    reserved_memory_bytes: int
    constrained_groups: tuple[Path, ...]


def _optional(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None


def _unescape(value: str) -> str:
    for source, target in (("\\040", " "), ("\\011", "\t"), ("\\134", "\\")):
        value = value.replace(source, target)
    return value


def current_cgroup() -> Cgroup:
    membership = next((line.split(":", 2)[2]
                       for line in Path("/proc/self/cgroup").read_text().splitlines()
                       if line.startswith("0::")), None)
    if membership is not None:
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            before, after = line.split(" - ", 1)
            if after.split()[0] == "cgroup2":
                fields = before.split()
                context = Cgroup(Path(_unescape(fields[3])), Path(_unescape(fields[4])), Path("/"))
                return Cgroup(context.mount_root, context.mount, context.locate(membership))
    raise RuntimeError("需要 cgroup v2 才能可靠限制内存并禁用本任务的 swap；未启动训练。")


def allocate_resources(topology: dict[tuple[int, int], set[int]], total_memory: int,
                       available_memory: int, cgroup: Cgroup) -> Resources:
    """Reserve whole physical cores; include headroom at every cgroup ancestor."""
    groups = sorted(topology.values(), key=lambda cpus: (-len(cpus), min(cpus)))
    if not groups:
        raise RuntimeError("当前账户没有可用 CPU。")
    reserve_count = 2 if len(groups) >= 6 else (1 if len(groups) >= 2 else 0)
    reserved = set().union(*groups[:reserve_count]) if reserve_count else set()
    cpus = sorted(set().union(*groups) - reserved)
    available, capacity, quota = available_memory, total_memory, math.inf
    constrained = []
    for node in cgroup.ancestors():
        usage = _optional(node / "memory.current")
        limited = False
        for filename in ("memory.max", "memory.high"):
            limit = _optional(node / filename)
            if limit is not None and limit != "max":
                if usage is None:
                    raise RuntimeError(f"无法读取内存用量：{node}")
                capacity = min(capacity, int(limit))
                available = min(available, max(0, int(limit) - int(usage)))
                limited = True
        value = _optional(node / "cpu.max")
        if value:
            amount, period = value.split()
            if amount != "max":
                quota = min(quota, int(amount) / int(period))
                limited = True
        if limited:
            constrained.append(node)
    reserve_memory = math.ceil(available * 0.25)
    if capacity >= 16 * GIB:
        reserve_memory = max(8 * GIB, reserve_memory)
    budget = (available - reserve_memory) // (1024**2) * (1024**2)
    if not cpus or budget <= 0:
        raise RuntimeError("预留交互资源后没有足够的 CPU 或内存；不会缩减训练配置。")
    threads = len(groups) - reserve_count
    if math.isfinite(quota):
        threads = min(threads, max(1, math.ceil(quota)))
    return Resources(tuple(cpus), tuple(sorted(reserved)), reserve_count, threads,
                     budget, available, reserve_memory, tuple(constrained))


def inspect_resources() -> tuple[Cgroup, Resources]:
    if sys.platform != "linux":
        raise RuntimeError("此后台资源启动器仅支持 Linux；未启动训练。")
    topology: dict[tuple[int, int], set[int]] = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        directory = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = (int((directory / "physical_package_id").read_text()),
               int((directory / "core_id").read_text()))
        topology.setdefault(key, set()).add(cpu)
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        memory[key] = int(value.split()[0]) * 1024
    if "MemAvailable" not in memory:
        raise RuntimeError("无法读取实际可用内存；未启动训练。")
    cgroup = current_cgroup()
    return cgroup, allocate_resources(topology, memory["MemTotal"], memory["MemAvailable"], cgroup)


def _query(command: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(command, text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None


def choose_backend(cgroup: Cgroup, resources: Resources, environment: dict[str, str]) -> str:
    scheduler = any(environment.get(key) for key in
                    ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "JOB_ID"))
    if not scheduler and shutil.which("systemd-run") and shutil.which("systemctl"):
        bus = _query(["systemctl", "--user", "show-environment"])
        manager = _query(["systemctl", "show", f"user@{os.getuid()}.service",
                          "--property=ControlGroup", "--value"])
        if bus is not None and bus.returncode == 0 and manager is not None and manager.returncode == 0:
            try:
                manager_path = cgroup.locate(manager.stdout.strip())
                if all(manager_path.is_relative_to(node) for node in resources.constrained_groups):
                    return "systemd"
            except RuntimeError:
                pass
    controllers = (_optional(cgroup.current / "cgroup.subtree_control") or "").split()
    if "memory" in controllers and os.access(cgroup.current, os.W_OK):
        return "detached"
    raise RuntimeError("没有可用的独立用户服务或已委派的 memory 控制器；无法保证内存预留，未启动训练。")


def training_command(data_dir: Path, checkpoint: Path, output: Path, uv: str,
                     *, ppo_minibatches: int | None = None, stage: str = "all") -> list[str]:
    def relative(path: Path) -> str:
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    if stage not in ("all", "base"):
        raise ValueError("stage must be all or base")
    if ppo_minibatches is not None and (type(ppo_minibatches) is not int or ppo_minibatches < 1):
        raise ValueError("PPO minibatch count must be a positive integer")

    captures = {}
    for default_split, manifests in (
            ("train", (data_dir / "capture").rglob("metadata.json")),
            ("validation", (data_dir / "validation").rglob("metadata.json"))):
        for manifest_path in manifests:
            directory = manifest_path.parent
            sidecar = directory / "capture.json"
            identity = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            split = identity.get("split", default_split)
            if split not in ("train", "validation"):
                raise ValueError(f"unknown capture split: {directory}")
            key = split, tuple(identity.get("episode_ids") or [relative(directory)])
            count = json.loads(manifest_path.read_text())["transition_count"]
            if key not in captures or count > captures[key][0]:
                captures[key] = count, relative(directory)
    train = sorted(value[1] for key, value in captures.items() if key[0] == "train")
    validation = sorted(value[1] for key, value in captures.items() if key[0] == "validation")
    if not train or not validation:
        raise RuntimeError("指定目录中缺少已采集的训练集或验证集；不会重新采集初始数据。")
    if not checkpoint.is_file():
        raise RuntimeError(f"策略 checkpoint 不存在：{checkpoint}")
    # All model/environment/schedule settings remain owned by latent_pretraining.yaml.
    return [uv, "run", "--no-sync", "scripts/pretrain_latent_path_following.py",
            f"policy_checkpoint={json.dumps(relative(checkpoint))}",
            f"output_dir={json.dumps(str(output))}", f"stage={stage}", "device=cuda", "capture.initial=false",
            "wm.data.train_dirs=" + json.dumps(train, separators=(",", ":")),
            "wm.data.validation_dirs=" + json.dumps(validation, separators=(",", ":")),
            "wm.training.microbatch_size=256", "capture.residual_amplitude_rad=0.005",
            "capture.risk_tilt_limit_rad=0.8",
            "capture.command_ranges=[[-0.6,1.0],[-0.4,0.4],[-0.8,0.8]]",
            "actor.training.residual_limit_rad=" + json.dumps([0.005] * 29),
            *([] if ppo_minibatches is None else [f"ppo.num_mini_batches={ppo_minibatches}"])]


def worker(payload: str, command: list[str]) -> None:
    if sys.platform != "linux":
        raise RuntimeError("训练 worker 仅支持 Linux。")
    settings = json.loads(payload)
    if settings["fallback"]:
        group = Path(settings["fallback"])
        (group / "cgroup.procs").write_text(str(os.getpid()))
    else:
        group = current_cgroup().current
    limit = (group / "memory.max").read_text().strip()
    swap = (group / "memory.swap.max").read_text().strip()
    if limit == "max" or int(limit) > settings["memory_bytes"] or swap != "0":
        raise RuntimeError("内存硬限制或禁用 swap 未生效，拒绝启动训练。")
    os.sched_setaffinity(0, settings["cpus"])
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.setpriority(os.PRIO_PROCESS, 0, 10)
    os.chdir(ROOT)
    os.execvpe(command[0], command, os.environ)


def launch(command: list[str], output: Path, cgroup: Cgroup, resources: Resources,
           backend: str, environment: dict[str, str], uv: str, ionice: str) -> str:
    log = Path(str(output) + ".log")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"新训练输出目录已存在：{output}")
    settings = {"cpus": resources.cpus, "memory_bytes": resources.memory_bytes, "fallback": None}
    group = None
    if backend == "detached":
        group = cgroup.current / f"sttt-latent-{output.name}-{os.getpid()}"
        group.mkdir()
        try:
            (group / "memory.max").write_text(str(resources.memory_bytes))
            (group / "memory.swap.max").write_text("0")
        except OSError:
            group.rmdir()
            raise
        settings["fallback"] = str(group)
    guarded = [uv, "run", "--no-sync", str(Path(__file__).resolve()), "--_worker",
               json.dumps(settings), ionice, "-c", "2", "-n", "7", *command]
    try:
        with log.open("xb", buffering=0) as handle:
            if backend == "systemd":
                unit = f"sttt-latent-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
                args = ["systemd-run", "--user", f"--unit={unit}", "--collect",
                        f"--working-directory={ROOT}",
                        f"--property=MemoryMax={resources.memory_bytes}",
                        "--property=MemorySwapMax=0", "--property=KillMode=control-group",
                        f"--property=StandardOutput=append:{log}",
                        f"--property=StandardError=append:{log}"]
                for key, value in environment.items():
                    if key in {"PATH", "HOME", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "PYTHONPATH",
                               "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                               "PYTHONUNBUFFERED", "PYTHONHASHSEED", "UV_CACHE_DIR"} or key.startswith(("CUDA_", "NVIDIA_")):
                        args.append(f"--setenv={key}={value}")
                unset = [key for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES")
                         if key not in environment]
                if unset:
                    args.append("--property=UnsetEnvironment=" + " ".join(unset))
                subprocess.run([*args, "--", *guarded], check=True)
                return f"后台服务：{unit}.service"
            process = subprocess.Popen(guarded, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            return f"后台进程：PID={process.pid}；资源组：{group}"
    except Exception:
        if group is not None:
            group.rmdir()
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="复用已采轨迹，按服务器资源预算后台启动既有训练流程。")
    parser.add_argument("--data-dir", type=Path, required=True, help="包含 capture/nominal 和 validation 的旧运行目录")
    parser.add_argument("--policy-checkpoint", type=Path, default=Path("model/mixup_0kg/student_iter_0031.pt"))
    parser.add_argument("--output-dir", type=Path, help="新运行目录；默认自动生成时间戳")
    parser.add_argument("--stage", choices=("all", "base"), default="all",
                        help="base 仅执行阶段1；all 依次执行阶段1和阶段2")
    parser.add_argument("--ppo-minibatches", type=int, help="每个 PPO epoch 的 minibatch 数；默认由训练 YAML 指定")
    parser.add_argument("--dry-run", action="store_true", help="仅显示资源分配，不创建输出文件或启动进程")
    options = parser.parse_args(argv)
    output = (options.output_dir or Path(f"outputs/latent_joint_reuse_{datetime.now():%Y%m%d_%H%M%S_%f}")).resolve()
    uv, ionice = shutil.which("uv"), shutil.which("ionice")
    if uv is None or ionice is None:
        raise RuntimeError("需要已安装的 uv 和 ionice；本脚本不安装工具。")
    command = training_command(options.data_dir.resolve(), options.policy_checkpoint.resolve(), output, uv,
                               ppo_minibatches=options.ppo_minibatches, stage=options.stage)
    cgroup, resources = inspect_resources()
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS=str(resources.threads), MKL_NUM_THREADS=str(resources.threads),
                       OPENBLAS_NUM_THREADS=str(resources.threads), PYTHONUNBUFFERED="1")
    backend = choose_backend(cgroup, resources, environment)
    plan = asdict(resources)
    plan["constrained_groups"] = [str(p) for p in resources.constrained_groups]
    print(json.dumps({"resources": plan, "backend": backend, "output": str(output),
                      "log": str(output) + ".log", "stage": options.stage, "dry_run": options.dry_run}, ensure_ascii=False), flush=True)
    print("nice≥10、I/O best-effort 7 是优先级；CPU 亲和范围和内存硬上限单独生效。", flush=True)
    if not resources.reserved_cores:
        print("当前仅有一颗可用物理核，无法另留完整核。", flush=True)
    if options.dry_run:
        return
    print(launch(command, output, cgroup, resources, backend, environment, uv, ionice), flush=True)
    print(f"查看日志：tail -f {str(output) + '.log'}", flush=True)


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--_worker":
            worker(sys.argv[2], sys.argv[3:])
        else:
            main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(f"启动失败：{error}") from error
