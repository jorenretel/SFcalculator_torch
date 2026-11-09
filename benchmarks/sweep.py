"""Compare dense and streamed calculations in fresh, sequential processes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.fprotein import positive_float, positive_int


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scales", nargs="+", type=positive_float, default=[1.0, 2.0])
    parser.add_argument(
        "--modes", nargs="+", choices=("forward", "coordinates", "all"), default=["coordinates"]
    )
    parser.add_argument("--chunks", nargs="+", type=positive_int, default=[2048, 8192])
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--timeout", type=positive_int, default=180)
    args = parser.parse_args()
    for name in ("scales", "modes", "chunks"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"--{name} must not contain duplicates")
    if args.output_dir.exists():
        parser.error("Sweep output directory already exists; choose a fresh one")
    args.output_dir.mkdir(parents=True)
    configurations = [("dense", 2048)] + [("streamed", chunk) for chunk in args.chunks]
    rows = []
    failed = False
    source_hashes = None
    for scale in args.scales:
        for mode in args.modes:
            for backend, chunk in configurations:
                filename = f"scale-{scale:g}-{mode}-{backend}-{chunk}.json"
                path = args.output_dir / filename
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.fprotein",
                    "--device",
                    args.device,
                    "--backend",
                    backend,
                    "--chunk-size",
                    str(chunk),
                    "--mode",
                    mode,
                    "--full-reflections",
                    "--resolution",
                    "3",
                    "--cell-scale",
                    str(scale),
                    "--repeats",
                    str(args.repeats),
                    "--saved-tensors",
                    "--output",
                    str(path),
                ]
                print(f"Starting {filename}", flush=True)
                try:
                    completed = subprocess.run(command, timeout=args.timeout, check=False)
                    if not path.exists():
                        raise RuntimeError(
                            f"Benchmark exited {completed.returncode} without a report"
                        )
                    report = json.loads(path.read_text())
                    status = report["status"]
                    expected_exit = {"ok": 0, "oom": 2}.get(status)
                    if expected_exit is None or completed.returncode != expected_exit:
                        raise RuntimeError("Unexpected benchmark failure")
                    hashes = report["code"]["source_sha256"]
                    if source_hashes is not None and hashes != source_hashes:
                        raise RuntimeError(
                            "Source changed during the sweep; measurements are not comparable"
                        )
                    source_hashes = hashes
                    summary = report.get("summary", {})
                    row = {
                        "file": filename,
                        "scale": scale,
                        "mode": mode,
                        "backend": backend,
                        "chunk_size": chunk,
                        "status": status,
                        "reflections": report["input"].get("asu_reflections"),
                        "forward_ms": summary.get(
                            "forward_cuda_ms", summary.get("forward_wall_ms", {})
                        ).get("median"),
                        "backward_ms": summary.get(
                            "backward_cuda_ms", summary.get("backward_wall_ms", {})
                        ).get("median"),
                        "peak_allocated_bytes": summary.get("peak_allocated_bytes"),
                        "saved_storage_bytes": report.get("saved_tensors", {}).get(
                            "unique_storage_bytes"
                        ),
                        "oom_operation": report.get("operation"),
                    }
                except subprocess.TimeoutExpired:
                    row = {"file": filename, "status": "timeout"}
                    failed = True
                except (RuntimeError, ValueError, KeyError) as error:
                    print(str(error), file=sys.stderr, flush=True)
                    row = {"file": filename, "status": "error"}
                    failed = True
                rows.append(row)
                print(json.dumps(row), flush=True)
                (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
