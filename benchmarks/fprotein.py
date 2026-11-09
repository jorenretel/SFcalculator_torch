"""Measure the public structure-factor API in an isolated CLI process.

Run from the checkout root: python -m benchmarks.fprotein --help
No model downloads, tracking services, or additional benchmark packages are used.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import gemmi
import numpy as np
import SFC_Torch
import torch
from tests.fprotein_reference import (
    CASES,
    load_case,
    make_calculator,
    parameter_tensors,
    projected_loss,
)

ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="cryo_1lj5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("forward", "coordinates", "all"), default="coordinates")
    parser.add_argument("--backend", choices=("streamed", "dense", "compiled"), default="streamed")
    parser.add_argument("--chunk-size", type=positive_int, default=2048)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument(
        "--full-reflections", action="store_true", help="Generate a full reciprocal ASU"
    )
    parser.add_argument(
        "--resolution", type=positive_float, default=3.0, help="Full-set dmin in Angstrom"
    )
    parser.add_argument(
        "--cell-scale",
        type=positive_float,
        default=1.0,
        help="Synthetic box expansion; coordinates stay fixed",
    )
    parser.add_argument(
        "--saved-tensors", action="store_true", help="Extra untimed autograd-storage inspection"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Extra untimed iteration with a CUDA kernel summary"
    )
    parser.add_argument(
        "--trace", type=Path, help="Extra untimed profiler iteration exported as a Chrome trace"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="JSON output; must not already exist"
    )
    return parser


def inputs(args):
    data = load_case(args.case)
    if args.cell_scale != 1.0 and not args.full_reflections:
        raise ValueError("--cell-scale requires --full-reflections")
    if args.full_reflections:
        data["cell"] = data["cell"].copy()
        data["cell"][:3] *= args.cell_scale
        data["hkl"] = gemmi.make_miller_array(
            gemmi.UnitCell(*data["cell"]),
            gemmi.SpaceGroup(str(data["spacegroup"])),
            args.resolution,
            0.0,
            unique=True,
        )
        if bool(data["anomalous"]):
            data["hkl"] = np.concatenate((data["hkl"], -data["hkl"]))
        rng = np.random.default_rng(args.seed)
        data["hkl"] = data["hkl"][rng.permutation(len(data["hkl"]))].copy()
    return data


def provenance():
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "--no-pager", *args],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("--no-optional-locks", "status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git("rev-parse", "HEAD"),
        "tracked_changes": None if status is None else bool(status),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "SFC_Torch").glob("*.py"))
            + [
                Path(__file__),
                ROOT / "tests" / "fprotein_reference.py",
                ROOT / "tests" / "data" / "fprotein_reference" / "manifest.json",
            ]
        },
    }


class Meter:
    def __init__(self, device):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def memory(self):
        if not self.cuda:
            return {"allocated_bytes": None, "reserved_bytes": None}
        return {
            "allocated_bytes": torch.cuda.memory_allocated(self.device),
            "reserved_bytes": torch.cuda.memory_reserved(self.device),
        }

    def reset_peak(self):
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak(self):
        if not self.cuda:
            return {"allocated_bytes": None, "reserved_bytes": None}
        return {
            "allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        }

    def timed(self, function):
        self.sync()
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        start_time = time.perf_counter()
        result = function()
        if self.cuda:
            end.record()
        self.sync()
        return result, {
            "wall_ms": (time.perf_counter() - start_time) * 1000,
            "cuda_ms": start.elapsed_time(end) if self.cuda else None,
        }


class SavedTensors:
    """Count saves without holding extra tensor references or copies."""

    def __init__(self):
        self.logical_bytes = 0
        self.storage = {}
        self.count = 0

    def pack(self, tensor):
        self.count += 1
        self.logical_bytes += tensor.numel() * tensor.element_size()
        storage = tensor.untyped_storage()
        self.storage[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        return tensor

    def unpack(self, tensor):
        return tensor

    def report(self):
        return {
            "save_calls": self.count,
            "logical_bytes": self.logical_bytes,
            "unique_storage_bytes": sum(self.storage.values()),
        }


def iteration(calculator, parameters, probe, mode, meter, active):
    # The calculator owns the previous output: drop it before the next iteration
    # so a prior graph/output cannot contaminate the next measurement.
    for attribute in ("Fprotein_asu", "Fprotein_HKL", "Fmask_asu", "Fmask_HKL"):
        calculator.__dict__.pop(attribute, None)
    for parameter in parameters.values():
        parameter.grad = None
    meter.sync()
    before = meter.memory()
    meter.reset_peak()
    active[0] = "forward"
    with torch.no_grad() if mode == "forward" else nullcontext():
        output, forward_time = meter.timed(lambda: calculator.calc_fprotein(Return=True))
    after_forward = meter.memory()
    forward_peak = meter.peak()
    backward_time = None
    backward_peak = None
    if mode != "forward":
        meter.reset_peak()
        active[0] = "loss_projection"
        loss = projected_loss(output, probe)
        active[0] = "backward"
        _, backward_time = meter.timed(loss.backward)
        backward_peak = meter.peak()
        del loss
    active[0] = "validation"
    if not torch.isfinite(output).all().item():
        raise ValueError("Non-finite structure factors")
    for parameter in parameters.values():
        if parameter.requires_grad and (
            parameter.grad is None or not torch.isfinite(parameter.grad).all().item()
        ):
            raise ValueError("Missing or non-finite gradient")
    peaks = [forward_peak] + ([backward_peak] if backward_peak else [])
    peak_allocated = max(p["allocated_bytes"] for p in peaks) if meter.cuda else None
    result = {
        "before": before,
        "after_forward": after_forward,
        "forward_peak": forward_peak,
        "backward_peak": backward_peak,
        "peak_allocated_bytes": peak_allocated,
        "incremental_peak_allocated_bytes": peak_allocated - before["allocated_bytes"]
        if meter.cuda
        else None,
        "forward": forward_time,
        "backward": backward_time,
    }
    del output
    return result


def summarize(samples):
    summary = {}
    for phase in ("forward", "backward"):
        for clock in ("wall_ms", "cuda_ms"):
            values = [
                sample[phase][clock]
                for sample in samples
                if sample[phase] is not None and sample[phase][clock] is not None
            ]
            if values:
                summary[f"{phase}_{clock}"] = {
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }
    for key in ("peak_allocated_bytes", "incremental_peak_allocated_bytes"):
        values = [sample[key] for sample in samples if sample[key] is not None]
        summary[key] = max(values) if values else None
    return summary


def kernel_summary(profiler):
    kernels = {}
    for event in profiler.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        entry = kernels.setdefault(event.name, {"name": event.name, "calls": 0, "total_us": 0.0})
        entry["calls"] += 1
        entry["total_us"] += event.time_range.elapsed_us()
    return {
        "scope": "Extra post-measurement iteration, including loss projection and validation",
        "note": (
            "CUDA device events include copies/memsets when present. Names/counts aid fusion "
            "inspection; they do not prove fusion or speedup."
        ),
        "cuda_kernel_calls": sum(entry["calls"] for entry in kernels.values()),
        "cuda_kernel_total_us": sum(entry["total_us"] for entry in kernels.values()),
        "kernels": sorted(kernels.values(), key=lambda entry: entry["total_us"], reverse=True),
    }


def run_benchmark(args):
    if Path(SFC_Torch.__file__).resolve().parent != ROOT / "SFC_Torch":
        raise ValueError("Benchmark must import the source checkout, not an installed wheel")
    meter = Meter(args.device)
    if meter.device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA are supported")
    if args.backend == "compiled" and not meter.cuda:
        raise ValueError("The compiled backend requires CUDA; no eager CPU fallback is benchmarked")
    if meter.cuda and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu only for a harness smoke test")
    data = inputs(args)
    report = {
        "schema_version": 1,
        "status": "running",
        "phase": "initialization",
        "code": provenance(),
        "settings": {
            key: value for key, value in vars(args).items() if key not in ("output", "trace")
        },
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "numpy": np.__version__,
            "gemmi": gemmi.__version__,
            "float32_matmul_precision": "highest",
        },
        "input": {
            "atoms": len(data["positions"]),
            "requested_reflections": len(data["hkl"]),
            "cell": data["cell"].tolist(),
            "mode": str(data["mode"]),
            "spacegroup": str(data["spacegroup"]),
            "dtype": "float32/complex64",
        },
        "measurement_notes": {
            "first_call": (
                "First warmup forward/backward in this process, including any lazy setup and JIT "
                "compilation for compiled. Cache-dependent, not a guaranteed cold compile; "
                "compiler caches are not cleared. Later calls may also specialize/recompile."
            ),
            "steady_state": "samples/summary exclude warmup and profiler/saved-tensor inspection",
            "timing": (
                "Wall time includes host work and synchronization. CUDA event intervals can "
                "include host/JIT idle gaps; they are not summed kernel execution time."
            ),
            "memory": (
                "CUDA PyTorch allocator only; excludes compiler/host and non-PyTorch allocations. "
                "CPU memory is not measured; reserved memory may persist across phases."
            ),
        },
        "first_call": None,
        "warmup_samples": [],
        "samples": [],
    }
    active = ["initialization"]
    previous_threads = torch.get_num_threads()
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    device_context = torch.cuda.device(meter.device) if meter.cuda else nullcontext()
    try:
        with device_context:
            if meter.cuda:
                props = torch.cuda.get_device_properties(meter.device)
                report["environment"]["gpu"] = {
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "capability": [props.major, props.minor],
                }
                # Context initialization is not part of constructor timing.
                torch.cuda.init()
                free, total = torch.cuda.mem_get_info(meter.device)
                report["environment"]["gpu"]["free_memory_before_bytes"] = free
                report["environment"]["gpu"]["driver_total_memory_bytes"] = total
                report["environment"]["allocator_backend"] = (
                    torch.cuda.memory.get_allocator_backend()
                )
            meter.sync()
            report["initial_memory"] = meter.memory()
            meter.reset_peak()

            def initialize():
                calculator = make_calculator(data, args.device)
                calculator.fprotein_backend = args.backend
                calculator.reflection_chunk_size = args.chunk_size
                if args.backend == "dense":
                    # Include preparation of the backend's persistent dense
                    # table, rather than hiding that allocation in warmup.
                    _ = calculator.fullsf_tensor
                return calculator

            calculator, timing = meter.timed(initialize)
            report["initialization"] = {
                "storage_layout": "compact-plus-dense" if args.backend == "dense" else "compact",
                "time": timing,
                "live": meter.memory(),
                "peak": meter.peak(),
            }
            report["input"]["asu_reflections"] = len(calculator.Hasu_array)
            active[0] = "parameters"
            parameters = parameter_tensors(data, args.device, requires_grad=args.mode == "all")
            parameters["positions"].requires_grad_(args.mode != "forward")
            calculator.atom_pos_orth = parameters["positions"]
            calculator.atom_b_iso = parameters["b_iso"]
            calculator.atom_aniso_uw = parameters["aniso_u"]
            calculator.atom_occ = parameters["occupancy"]
            active[0] = "probe"
            rng = np.random.default_rng(args.seed)
            probe = torch.tensor(
                (
                    rng.normal(size=len(calculator.HKL_array))
                    + 1j * rng.normal(size=len(calculator.HKL_array))
                ).astype(np.complex64),
                device=args.device,
            )
            for _ in range(args.warmup):
                report["phase"] = "warmup"
                sample = iteration(calculator, parameters, probe, args.mode, meter, active)
                report["warmup_samples"].append(sample)
                if report["first_call"] is None:
                    report["first_call"] = sample
            report["warmup_summary"] = summarize(report["warmup_samples"])
            for _ in range(args.repeats):
                report["phase"] = "measurement"
                report["samples"].append(
                    iteration(calculator, parameters, probe, args.mode, meter, active)
                )
            report["summary"] = summarize(report["samples"])
            if args.saved_tensors and args.mode != "forward":
                report["phase"] = "saved_tensors"
                counter = SavedTensors()
                with torch.autograd.graph.saved_tensors_hooks(counter.pack, counter.unpack):
                    iteration(calculator, parameters, probe, args.mode, meter, active)
                report["saved_tensors"] = counter.report()
            if args.profile or args.trace:
                report["phase"] = "profiler"
                activities = [torch.profiler.ProfilerActivity.CPU]
                if meter.cuda:
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(
                    activities=activities, profile_memory=True, record_shapes=True
                ) as profiler:
                    iteration(calculator, parameters, probe, args.mode, meter, active)
                report["profiler"] = kernel_summary(profiler)
                if args.trace:
                    profiler.export_chrome_trace(str(args.trace))
                    trace = json.loads(args.trace.read_text())
                    for key in ("host_name", "traceName", "deviceProperties", "distributedInfo"):
                        trace.pop(key, None)
                    args.trace.write_text(json.dumps(trace))
            report["status"] = "ok"
            report.pop("phase", None)
    except torch.OutOfMemoryError:
        report["status"] = "oom"
        report["operation"] = active[0]
        report["oom_peak"] = meter.peak()
        report["warmup_summary"] = summarize(report["warmup_samples"])
        report["summary"] = summarize(report["samples"])
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        torch.set_num_threads(previous_threads)
    return report


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.output.exists() or (args.trace and args.trace.exists()):
        parser.error("Refusing to overwrite an existing report or trace")
    if args.trace == args.output:
        parser.error("Report and trace must have different paths")
    try:
        for path in (args.output, args.trace):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.error(f"Cannot prepare output directory: {error}")
    try:
        result = run_benchmark(args)
    except ValueError as error:
        parser.error(str(error))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "summary": result.get("summary")}, indent=2))
    raise SystemExit(0 if result["status"] == "ok" else 2)


if __name__ == "__main__":
    main()
