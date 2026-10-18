#!/usr/bin/env python3
"""Sample system and Ollama resource usage during one measured run."""

import argparse
import csv
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import pynvml


CSV_COLUMNS = [
    "timestamp_utc",
    "monotonic_s",
    "cpu_utilization_pct",
    "cpu_temperature_c",
    "system_ram_used_mb",
    "ollama_rss_mb",
    "gpu_utilization_pct",
    "gpu_memory_used_mb",
    "gpu_power_w",
    "gpu_temperature_c",
    "gpu_energy_counter_mj",
]

RUNNING = True


def stop_sampler(_signum, _frame) -> None:
    global RUNNING
    RUNNING = False


def collect_cpu_metrics() -> dict[str, float]:
    rss_bytes = 0
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            name = process.info["name"] or ""
            command = " ".join(process.info["cmdline"] or [])
            if "ollama" in name.lower() or "ollama" in command.lower():
                rss_bytes += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    memory = psutil.virtual_memory()
    temperatures = psutil.sensors_temperatures(fahrenheit=False)["coretemp"]
    package = next(sensor for sensor in temperatures if sensor.label.startswith("Package id"))

    return {
        "cpu_utilization_pct": psutil.cpu_percent(interval=None),
        "cpu_temperature_c": float(package.current),
        "system_ram_used_mb": memory.used / (1024 * 1024),
        "ollama_rss_mb": rss_bytes / (1024 * 1024),
    }


def collect_gpu_metrics(handle) -> dict[str, float]:
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

    return {
        "gpu_utilization_pct": utilization.gpu,
        "gpu_memory_used_mb": memory.used / (1024 * 1024),
        "gpu_power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
        "gpu_temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        "gpu_energy_counter_mj": pynvml.nvmlDeviceGetTotalEnergyConsumption(handle),
    }


def collect_samples(output_path: Path, interval_ms: int, gpu_handle) -> None:
    interval_s = interval_ms / 1000
    next_sample_at = time.monotonic()
    psutil.cpu_percent(interval=None)

    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        while RUNNING:
            writer.writerow(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "monotonic_s": time.monotonic(),
                    **collect_cpu_metrics(),
                    **collect_gpu_metrics(gpu_handle),
                }
            )
            destination.flush()

            next_sample_at += interval_s
            time.sleep(max(0, next_sample_at - time.monotonic()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=100)
    args = parser.parse_args()

    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop_sampler)
    signal.signal(signal.SIGINT, stop_sampler)

    pynvml.nvmlInit()
    try:
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        collect_samples(args.output, args.interval_ms, gpu_handle)
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
