#!/usr/bin/env python3
"""Render matched CPU, KORA/Vulkan, and raw Vulkan vector-add rows from a kora-bench report."""

import argparse
import json
import re
from pathlib import Path


def format_ns(value: int) -> str:
    if value < 1_000:
        return f"{value} ns"
    return f"{value / 1_000:,.3f} µs" if value < 1_000_000 else f"{value / 1_000_000:,.3f} ms"


def _nearest_rank(values: list[int], percentile: int) -> int:
    ordered = sorted(values)
    rank = max(1, (percentile * len(ordered) + 99) // 100)
    return ordered[rank - 1]


def validate_cuda_report(kora_report: dict, cuda_report: dict) -> list[dict]:
    """Validate optional externally produced CUDA measurements against one KORA report."""
    if not isinstance(kora_report, dict) or not isinstance(cuda_report, dict):
        raise ValueError("KORA and CUDA reports must be JSON objects")
    if kora_report.get("schema_version") != 3:
        raise ValueError("KORA report must use schema version 3")
    if kora_report.get("diagnostic_profile") is True:
        raise ValueError("diagnostic profile reports include telemetry overhead")
    kora_gpu = kora_report.get("gpu_measurements")
    if not isinstance(kora_gpu, dict) or not kora_gpu.get("same_physical_device_for_kora_and_raw_vulkan"):
        raise ValueError("KORA and raw Vulkan must identify the same physical device")
    if cuda_report.get("schema_version") != 1 or cuda_report.get("backend") != "cuda":
        raise ValueError("CUDA reference must use schema version 1 and backend 'cuda'")
    environment = cuda_report.get("environment")
    device = cuda_report.get("device")
    if not isinstance(environment, dict) or not all(
        isinstance(environment.get(key), str) and environment[key].strip()
        for key in ("os", "architecture", "compiler", "build_type", "source_revision")
    ):
        raise ValueError("CUDA environment must include OS, architecture, compiler, build type, and source revision")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", environment["source_revision"]):
        raise ValueError("CUDA source_revision must be a Git commit hash")
    if not isinstance(device, dict) or not all(
        isinstance(device.get(key), str) and device[key].strip()
        for key in ("name", "driver_version", "runtime_version")
    ):
        raise ValueError("CUDA device must include name, driver version, and runtime version")
    match = cuda_report.get("device_match")
    if not isinstance(match, dict) or match.get("same_physical_device_as_kora_vulkan") is not True:
        raise ValueError("CUDA comparison requires a verified same-physical-device match")
    if match.get("verification_method") != "cuda_uuid_equals_vk_device_uuid":
        raise ValueError("device match must be verified by comparing CUDA and Vulkan UUIDs")
    if not isinstance(cuda_report.get("results"), list) or not cuda_report["results"]:
        raise ValueError("CUDA reference must include at least one result")

    kora_result_items = kora_report.get("results")
    if not isinstance(kora_result_items, list) or not all(
        isinstance(result, dict) for result in kora_result_items
    ):
        raise ValueError("KORA report results must be a list of objects")
    kora_results = {result.get("name"): result for result in kora_result_items}
    validated = []
    seen = set()
    for item in cuda_report["results"]:
        if not isinstance(item, dict):
            raise ValueError("each CUDA result must be an object")
        required_strings = (
            "workload_id",
            "precision",
            "semantic_contract",
            "comparison_class",
            "implementation",
            "measurement_boundary",
            "correctness_method",
            "kora_result_name",
            "raw_vulkan_result_name",
        )
        if not all(isinstance(item.get(key), str) and item[key].strip() for key in required_strings):
            raise ValueError("CUDA result is missing workload, precision, semantics, class, or implementation")
        if item["comparison_class"] not in {"algorithm_matched", "best_in_class_library"}:
            raise ValueError("comparison_class must be algorithm_matched or best_in_class_library")
        if item.get("timing_domain") != "gpu_execution_event" or item["measurement_boundary"] != "kernel_dispatch":
            raise ValueError("CUDA result must report GPU event timing around the kernel dispatch")
        if item.get("unit") != "ns":
            raise ValueError("CUDA result unit must be ns")
        if item.get("correctness_checked") is not True or item.get("correctness_passed") is not True:
            raise ValueError("CUDA result must pass a correctness check before timing")
        size = item.get("size")
        iterations = item.get("iterations")
        warmup = item.get("warmup")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError("CUDA result size must be a positive integer")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 2:
            raise ValueError("CUDA result must include at least two measured iterations")
        if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
            raise ValueError("CUDA result warmup must be a nonnegative integer")
        samples = item.get("samples")
        if not isinstance(samples, list) or len(samples) != iterations or not all(
            isinstance(sample, int) and not isinstance(sample, bool) and sample > 0 for sample in samples
        ):
            raise ValueError("CUDA raw samples must contain one positive integer nanosecond value per iteration")
        for field, percentile in (("median", 50), ("p95", 95)):
            value = item.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value != _nearest_rank(samples, percentile):
                raise ValueError(f"CUDA {field} must equal the nearest-rank summary of its raw samples")
        identity = (item["workload_id"], size, item["comparison_class"], item["implementation"])
        if identity in seen:
            raise ValueError("CUDA result workload/class/implementation records must be unique")
        seen.add(identity)

        kora_name = item["kora_result_name"]
        raw_name = item["raw_vulkan_result_name"]
        token = f"_{item['workload_id']}_{size}_"
        if (
            not kora_name.startswith("gpu_")
            or token not in kora_name
            or not kora_name.endswith("_gpu_execution")
        ):
            raise ValueError("kora_result_name must identify a matching KORA GPU timestamp workload and size")
        if (
            not raw_name.startswith("raw_vulkan_")
            or token not in raw_name
            or not raw_name.endswith("_gpu_execution")
        ):
            raise ValueError("raw_vulkan_result_name must identify a matching raw Vulkan timestamp workload and size")
        kora_timing = kora_results.get(kora_name)
        raw_timing = kora_results.get(raw_name)
        for name, result in ((kora_name, kora_timing), (raw_name, raw_timing)):
            if result is not None and result.get("timing_domain") != "gpu_execution_timestamp":
                raise ValueError(f"{name} must use gpu_execution_timestamp")
        validated.append(
            {
                **item,
                "kora_median": kora_timing.get("median") if kora_timing else None,
                "kora_p95": kora_timing.get("p95") if kora_timing else None,
                "raw_vulkan_median": raw_timing.get("median") if raw_timing else None,
                "raw_vulkan_p95": raw_timing.get("p95") if raw_timing else None,
            }
        )
    return sorted(validated, key=lambda item: (item["workload_id"], item["size"], item["comparison_class"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="schema-3 kora-bench JSON report")
    parser.add_argument("--cuda-report", type=Path, help="optional version-1 external CUDA reference report")
    parser.add_argument("--kora-revision", help="Git commit hash of the KORA build (required with --cuda-report)")
    parser.add_argument("--sizes", nargs="+", type=int, default=[4096, 1048576, 4194240])
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("schema_version") != 3:
        parser.error("expected kora-bench JSON schema version 3")
    if report.get("diagnostic_profile") is True:
        parser.error("diagnostic profile reports include telemetry overhead and are not comparison tables")
    if "gpu_measurements" not in report:
        parser.error("report must include --gpu measurements")
    if not report["gpu_measurements"].get("same_physical_device_for_kora_and_raw_vulkan"):
        parser.error("KORA and raw Vulkan must identify the same physical device")
    results = {item["name"]: item for item in report["results"]}

    print("# Benchmark comparison\n")
    print(
        "| Elements | CPU median | CPU p95 | KORA default median | KORA default p95 | "
        "KORA deferred median | KORA deferred p95 | Raw host median | Raw host p95 | "
        "KORA GPU median | KORA GPU p95 | Raw GPU median | Raw GPU p95 |"
    )
    print("| " + " | ".join(["---:"] * 13) + " |")
    for size in args.sizes:
        names = (
            f"cpu_vector_add_{size}",
            f"gpu_vector_add_{size}_dispatch_wait",
            f"gpu_vector_add_{size}_deferred_readback_dispatch_wait",
            f"raw_vulkan_vector_add_{size}_dispatch_wait",
        )
        try:
            cpu, kora, kora_deferred, raw = (results[name] for name in names)
        except KeyError as error:
            parser.error(f"report is missing matched workload {error.args[0]}")
        gpu_names = (
            f"gpu_vector_add_{size}_gpu_execution",
            f"raw_vulkan_vector_add_{size}_gpu_execution",
        )
        gpu_results = [results.get(name) for name in gpu_names]
        values = [
            str(size),
            format_ns(cpu["median"]),
            format_ns(cpu["p95"]),
            format_ns(kora["median"]),
            format_ns(kora["p95"]),
            format_ns(kora_deferred["median"]),
            format_ns(kora_deferred["p95"]),
            format_ns(raw["median"]),
            format_ns(raw["p95"]),
            *[
                format_ns(result[stat]) if result is not None else "unavailable"
                for result in gpu_results
                for stat in ("median", "p95")
            ],
        ]
        print("| " + " | ".join(values) + " |")
    matched_compute_workloads = (
        (
            "SAXPY / 4,096",
            "gpu_saxpy_4096_upload_dispatch_wait",
            "raw_vulkan_saxpy_4096_upload_dispatch_wait",
            "gpu_saxpy_4096_gpu_execution",
            "raw_vulkan_saxpy_4096_upload_gpu_execution",
        ),
        (
            "Reduction / 4,096",
            "gpu_reduction_4096_dispatch_wait",
            "raw_vulkan_reduction_4096_dispatch_wait",
            "gpu_reduction_4096_gpu_execution",
            "raw_vulkan_reduction_4096_gpu_execution",
        ),
        (
            "Scan / 4,096",
            "gpu_tiled_scan_4096_dispatch_wait",
            "raw_vulkan_tiled_scan_4096_dispatch_wait",
            "gpu_tiled_scan_4096_gpu_execution",
            "raw_vulkan_tiled_scan_4096_gpu_execution",
        ),
        (
            "Tiled matmul / 64×64",
            "gpu_tiled_matmul_64_dispatch_wait",
            "raw_vulkan_tiled_matmul_64_dispatch_wait",
            "gpu_tiled_matmul_64_gpu_execution",
            "raw_vulkan_tiled_matmul_64_gpu_execution",
        ),
        (
            "Stencil / 4,096",
            "gpu_stencil_4096_dispatch_wait",
            "raw_vulkan_stencil_4096_dispatch_wait",
            "gpu_stencil_4096_gpu_execution",
            "raw_vulkan_stencil_4096_gpu_execution",
        ),
    )
    if all(kora_name in results and raw_name in results for _, kora_name, raw_name, _, _ in matched_compute_workloads):
        print("\nMatched same-device Vulkan compute workloads:")
        print(
            "| Workload | KORA host p50 | KORA host p95 | Raw Vulkan host p50 | Raw Vulkan host p95 | "
            "KORA GPU p50 | KORA GPU p95 | Raw Vulkan GPU p50 | Raw Vulkan GPU p95 |"
        )
        print("| " + " | ".join(["---"] + ["---:"] * 8) + " |")
        for label, kora_name, raw_name, kora_gpu_name, raw_gpu_name in matched_compute_workloads:
            kora_host = results[kora_name]
            raw_host = results[raw_name]
            if kora_host.get("timing_domain") != "host_submit_completion" or raw_host.get(
                "timing_domain"
            ) != "host_submit_completion":
                parser.error(f"{label} host timings must use host_submit_completion")
            gpu_results = (results.get(kora_gpu_name), results.get(raw_gpu_name))
            for gpu_result in gpu_results:
                if gpu_result is not None and gpu_result.get("timing_domain") != "gpu_execution_timestamp":
                    parser.error(f"{label} device timings must use gpu_execution_timestamp")
            values = [
                label,
                format_ns(kora_host["median"]),
                format_ns(kora_host["p95"]),
                format_ns(raw_host["median"]),
                format_ns(raw_host["p95"]),
                *[
                    format_ns(gpu_result[stat]) if gpu_result is not None else "unavailable"
                    for gpu_result in gpu_results
                    for stat in ("median", "p95")
                ],
            ]
            print("| " + " | ".join(values) + " |")
        print(
            "\nHost rows include submit through completion and are not kernel-only. Device-timestamp rows "
            "isolate execution; unavailable samples are not estimated. Raw Vulkan uses persistent "
            "host-visible buffers on this benchmark path, so its host latency has a different "
            "residency/readback boundary from KORA."
        )
    if args.cuda_report:
        if not args.kora_revision or not re.fullmatch(r"[0-9a-fA-F]{7,40}", args.kora_revision):
            parser.error("--cuda-report requires --kora-revision with a Git commit hash")
        try:
            cuda_report = json.loads(args.cuda_report.read_text(encoding="utf-8"))
            cuda_rows = validate_cuda_report(report, cuda_report)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            parser.error(str(error))
        print("\nOptional same-device CUDA reference (GPU event time; not host API latency):")
        print(
            f"KORA source revision: {args.kora_revision}; "
            f"CUDA source revision: {cuda_report['environment']['source_revision']}"
        )
        print(
            "| Workload | Elements | Precision | Comparison class | Implementation | "
            "CUDA median | CUDA p95 | KORA Vulkan median | Raw Vulkan median | CUDA/KORA ratio |"
        )
        print("| " + " | ".join(["---:"] * 10) + " |")
        for item in cuda_rows:
            ratio = (
                f"{item['median'] / item['kora_median']:.3f}×"
                if item["kora_median"] is not None
                else "unavailable"
            )
            values = [
                item["workload_id"],
                f"{item['size']:,}",
                item["precision"],
                item["comparison_class"],
                item["implementation"],
                format_ns(item["median"]),
                format_ns(item["p95"]),
                format_ns(item["kora_median"]) if item["kora_median"] is not None else "unavailable",
                format_ns(item["raw_vulkan_median"])
                if item["raw_vulkan_median"] is not None
                else "unavailable",
                ratio,
            ]
            print("| " + " | ".join(values) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
