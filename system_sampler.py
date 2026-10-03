#!/usr/bin/env python3
"""Sample machine and Ollama-process resource use during one measured run."""

import argparse
import csv
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

try:
    import pynvml
except ImportError:  # The experiment can still collect CPU/RAM measurements.
    pynvml = None


RUNNING = True


def stop_sampler(_signum, _frame):
    global RUNNING
    RUNNING = False


def ollama_processes():
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            name = process.info["name"] or ""
            command = " ".join(process.info["cmdline"] or [])
            if "ollama" in name.lower() or "ollama" in command.lower():
                yield process
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def ollama_rss_mb():
    total = 0
    for process in ollama_processes():
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / (1024 * 1024)


def gpu_metrics():
    if pynvml is None:
        return {"gpu_utilization_pct": None, "gpu_memory_used_mb": None,
                "gpu_power_w": None, "gpu_temperature_c": None,
                "gpu_energy_counter_mj": None}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        try:
            energy_counter = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        except pynvml.NVMLError:
            energy_counter = None
        return {
            "gpu_utilization_pct": utilization.gpu,
            "gpu_memory_used_mb": memory.used / (1024 * 1024),
            "gpu_power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            "gpu_temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            "gpu_energy_counter_mj": energy_counter,
        }
    except pynvml.NVMLError:
        return {"gpu_utilization_pct": None, "gpu_memory_used_mb": None,
                "gpu_power_w": None, "gpu_temperature_c": None,
                "gpu_energy_counter_mj": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=100)
    args = parser.parse_args()
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop_sampler)
    signal.signal(signal.SIGINT, stop_sampler)
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            pass

    fieldnames = [
        "timestamp_utc", "monotonic_s", "cpu_utilization_pct", "system_ram_used_mb", "ollama_rss_mb",
        "gpu_utilization_pct", "gpu_memory_used_mb", "gpu_power_w", "gpu_temperature_c",
        "gpu_energy_counter_mj",
    ]
    psutil.cpu_percent(interval=None)  # Prime psutil's non-blocking measurement.
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        while RUNNING:
            memory = psutil.virtual_memory()
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "monotonic_s": time.monotonic(),
                "cpu_utilization_pct": psutil.cpu_percent(interval=None),
                "system_ram_used_mb": memory.used / (1024 * 1024),
                "ollama_rss_mb": ollama_rss_mb(),
                **gpu_metrics(),
            }
            writer.writerow(row)
            output.flush()
            time.sleep(args.interval_ms / 1000)
    if pynvml is not None:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


if __name__ == "__main__":
    main()
