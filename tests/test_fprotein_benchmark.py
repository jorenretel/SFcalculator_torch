"""CPU smoke tests for the benchmark protocol, not performance assertions."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from benchmarks import fprotein as benchmark


@pytest.fixture
def arguments(tmp_path):
    return benchmark.build_parser().parse_args(
        [
            "--case",
            "cryo_p1_iso",
            "--device",
            "cpu",
            "--repeats",
            "2",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )


@pytest.mark.parametrize("mode", ["forward", "coordinates", "all"])
def test_cpu_measurements(arguments, mode):
    arguments.mode = mode
    arguments.saved_tensors = True
    threads = torch.get_num_threads()
    precision = torch.get_float32_matmul_precision()
    result = benchmark.run_benchmark(arguments)
    assert result["status"] == "ok"
    assert torch.get_num_threads() == threads
    assert torch.get_float32_matmul_precision() == precision
    assert len(result["samples"]) == 2
    assert len(result["warmup_samples"]) == arguments.warmup
    assert result["first_call"] == result["warmup_samples"][0]
    assert result["warmup_summary"] == benchmark.summarize(result["warmup_samples"])
    assert result["summary"] == benchmark.summarize(result["samples"])
    assert "cache-dependent" in result["measurement_notes"]["first_call"].lower()
    assert result["input"]["atoms"] == 11
    assert result["input"]["asu_reflections"] == 257
    assert result["summary"]["peak_allocated_bytes"] is None
    for sample in result["warmup_samples"] + result["samples"]:
        assert sample["forward"]["wall_ms"] > 0
        assert sample["forward"]["cuda_ms"] is None
        assert sample["before"]["allocated_bytes"] is None
        assert (sample["backward"] is None) == (mode == "forward")
    if mode == "forward":
        assert "saved_tensors" not in result
    else:
        assert result["saved_tensors"]["save_calls"] > 0
        assert result["saved_tensors"]["unique_storage_bytes"] > 0
    assert "output" not in result["settings"]
    assert "trace" not in result["settings"]
    assert str(arguments.output.parent) not in json.dumps(result)
    assert all(not Path(name).is_absolute() for name in result["code"]["source_sha256"])


@pytest.mark.parametrize(
    "backend, expected",
    [(None, "streamed"), ("dense", "dense"), ("streamed", "streamed"), ("compiled", "compiled")],
)
def test_backend_option(backend, expected):
    command = ["--output", "unused.json"]
    if backend is not None:
        command.extend(["--backend", backend])
    assert benchmark.build_parser().parse_args(command).backend == expected


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_compiled_requires_available_cuda(arguments, monkeypatch, device):
    arguments.backend = "compiled"
    arguments.device = device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def must_not_load_inputs(args):
        pytest.fail("Compiled backend must reject unsupported devices before loading inputs")

    monkeypatch.setattr(benchmark, "inputs", must_not_load_inputs)
    message = "compiled backend requires CUDA" if device == "cpu" else "CUDA unavailable"
    with pytest.raises(ValueError, match=message):
        benchmark.run_benchmark(arguments)


@pytest.mark.parametrize("mode", ["forward", "coordinates", "all"])
def test_first_call_and_warmup_are_excluded_from_steady_state(arguments, monkeypatch, mode):
    arguments.mode = mode
    arguments.warmup = 2
    samples = [
        {
            "forward": {"wall_ms": duration, "cuda_ms": None},
            "backward": {"wall_ms": 2 * duration, "cuda_ms": None} if mode != "forward" else None,
            "peak_allocated_bytes": peak,
            "incremental_peak_allocated_bytes": peak - 100,
        }
        for duration, peak in [(100, 1000), (40, 800), (2, 200), (4, 300)]
    ]
    pending = iter(samples)
    monkeypatch.setattr(benchmark, "iteration", lambda *args: next(pending))
    result = benchmark.run_benchmark(arguments)
    assert result["status"] == "ok"
    assert result["first_call"] == samples[0]
    assert result["warmup_samples"] == samples[:2]
    assert result["samples"] == samples[2:]
    for key, durations, peak, incremental in (
        ("warmup_summary", (40, 100), 1000, 900),
        ("summary", (2, 4), 300, 200),
    ):
        summary = result[key]
        assert summary["forward_wall_ms"] == {
            "median": sum(durations) / 2,
            "min": durations[0],
            "max": durations[1],
        }
        if mode == "forward":
            assert "backward_wall_ms" not in summary
        else:
            assert summary["backward_wall_ms"] == {
                "median": sum(durations),
                "min": 2 * durations[0],
                "max": 2 * durations[1],
            }
        assert summary["peak_allocated_bytes"] == peak
        assert summary["incremental_peak_allocated_bytes"] == incremental
    assert next(pending, None) is None


def test_kernel_summary_aggregates_cuda_events():
    def event(name, duration):
        return SimpleNamespace(
            name=name,
            device_type=torch.autograd.DeviceType.CUDA,
            time_range=SimpleNamespace(elapsed_us=lambda: duration),
        )

    profiler = SimpleNamespace(
        events=lambda: [
            event("fused_tile", 3.0),
            SimpleNamespace(name="host_op", device_type=torch.autograd.DeviceType.CPU),
            event("other_kernel", 10.0),
            event("fused_tile", 5.0),
        ]
    )
    summary = benchmark.kernel_summary(profiler)
    assert summary["cuda_kernel_calls"] == 3
    assert summary["cuda_kernel_total_us"] == 18.0
    assert summary["kernels"] == [
        {"name": "other_kernel", "calls": 1, "total_us": 10.0},
        {"name": "fused_tile", "calls": 2, "total_us": 8.0},
    ]
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("cpu_only", [False, True])
def test_kernel_summary_without_cuda_events(cpu_only):
    events = [SimpleNamespace(device_type=torch.autograd.DeviceType.CPU)] if cpu_only else []
    summary = benchmark.kernel_summary(SimpleNamespace(events=lambda: events))
    assert summary["cuda_kernel_calls"] == 0
    assert summary["cuda_kernel_total_us"] == 0
    assert summary["kernels"] == []


def test_full_reflection_scaling(arguments):
    arguments.full_reflections = True
    original = benchmark.inputs(arguments)
    arguments.cell_scale = 2.0
    expanded = benchmark.inputs(arguments)
    np.testing.assert_array_equal(original["positions"], expanded["positions"])
    np.testing.assert_array_equal(expanded["cell"][:3], original["cell"][:3] * 2)
    assert len(expanded["hkl"]) > len(original["hkl"]) * 7
    assert len(np.unique(expanded["hkl"], axis=0)) == len(expanded["hkl"])
    np.testing.assert_array_equal(expanded["hkl"], benchmark.inputs(arguments)["hkl"])


def test_cell_scale_requires_full_set(arguments):
    arguments.cell_scale = 2.0
    with pytest.raises(ValueError, match="requires --full-reflections"):
        benchmark.inputs(arguments)


@pytest.mark.parametrize(
    "field", ["--repeats", "--warmup", "--threads", "--resolution", "--cell-scale"]
)
def test_positive_settings(field):
    with pytest.raises(SystemExit):
        benchmark.build_parser().parse_args([field, "0", "--output", "unused.json"])


def test_saved_tensor_storage_is_not_counted_twice():
    counter = benchmark.SavedTensors()
    value = torch.ones(4)
    assert counter.pack(value) is value
    counter.pack(value[:2])
    assert counter.report() == {"save_calls": 2, "logical_bytes": 24, "unique_storage_bytes": 16}


@pytest.mark.parametrize(
    "function, operation",
    [
        ("make_calculator", "initialization"),
        ("parameter_tensors", "parameters"),
        ("projected_loss", "loss_projection"),
    ],
)
def test_oom_is_reported(arguments, monkeypatch, function, operation):
    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("injected")

    monkeypatch.setattr(benchmark, function, fail)
    precision = torch.get_float32_matmul_precision()
    result = benchmark.run_benchmark(arguments)
    assert result["status"] == "oom"
    assert result["operation"] == operation
    assert torch.get_float32_matmul_precision() == precision


def test_partial_results_survive_oom(arguments, monkeypatch):
    original = benchmark.iteration
    calls = []

    def injected(*args):
        calls.append(None)
        if len(calls) == 3:  # warmup, one measured sample, then failure
            args[-1][0] = "backward"
            raise torch.OutOfMemoryError("injected")
        return original(*args)

    monkeypatch.setattr(benchmark, "iteration", injected)
    result = benchmark.run_benchmark(arguments)
    assert result["status"] == "oom"
    assert result["phase"] == "measurement"
    assert result["operation"] == "backward"
    assert len(result["samples"]) == 1
    assert result["summary"]["forward_wall_ms"]["median"] > 0


def test_cuda_timer_synchronizes_and_brackets_operation(monkeypatch):
    calls = []
    events = []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            self.index = len(events)
            events.append(self)

        def record(self):
            calls.append(f"event{self.index}")

        def elapsed_time(self, other):
            assert other is events[1]
            return 3.0

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))
    meter = benchmark.Meter("cuda:1")
    _, timing = meter.timed(lambda: calls.append("operation"))
    assert calls == ["sync", "event0", "operation", "event1", "sync"]
    assert timing["cuda_ms"] == 3.0
    assert timing["wall_ms"] > 0


def test_cli_json_and_overwrite_guard(arguments, monkeypatch):
    arguments.output = arguments.output.parent / "nested" / "reports" / "report.json"
    original = benchmark.run_benchmark

    def check_directory(args):
        assert args.output.parent.is_dir()
        return original(args)

    monkeypatch.setattr(benchmark, "run_benchmark", check_directory)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fprotein",
            "--case",
            arguments.case,
            "--device",
            "cpu",
            "--repeats",
            "1",
            "--output",
            str(arguments.output),
        ],
    )
    with pytest.raises(SystemExit) as completed:
        benchmark.main()
    assert completed.value.code == 0
    content = arguments.output.read_text()
    assert json.loads(content)["status"] == "ok"
    with pytest.raises(SystemExit) as refused:
        benchmark.main()
    assert refused.value.code == 2
    assert arguments.output.read_text() == content


def test_output_directory_failure_precedes_benchmark(arguments, monkeypatch, tmp_path, capsys):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("preserve")
    monkeypatch.setattr(sys, "argv", ["fprotein", "--output", str(blocker / "report.json")])

    def must_not_run(args):
        pytest.fail("Benchmark ran before output directory validation")

    monkeypatch.setattr(benchmark, "run_benchmark", must_not_run)
    with pytest.raises(SystemExit) as refused:
        benchmark.main()
    assert refused.value.code == 2
    assert "Cannot prepare output directory" in capsys.readouterr().err
    assert blocker.read_text() == "preserve"


def test_profiler_metadata_is_removed(arguments, monkeypatch, tmp_path):
    trace = tmp_path / "nested" / "traces" / "trace.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fprotein",
            "--case",
            arguments.case,
            "--device",
            "cpu",
            "--mode",
            "forward",
            "--repeats",
            "1",
            "--trace",
            str(trace),
            "--output",
            str(arguments.output),
        ],
    )
    with pytest.raises(SystemExit) as completed:
        benchmark.main()
    assert completed.value.code == 0
    data = json.loads(trace.read_text())
    assert data["traceEvents"]
    assert not {"host_name", "traceName", "deviceProperties", "distributedInfo"} & data.keys()
    assert str(tmp_path) not in trace.read_text()
