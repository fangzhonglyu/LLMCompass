# gpu_profile_iter.py
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage
import time, threading, argparse
from statistics import median
from pathlib import Path
from matplotlib import pyplot as plt

# --- your software model imports ---
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU
from software_model.matmul import Matmul
from software_model.softmax import Softmax
from software_model.utils import data_type_dict, Tensor

# =========================
# NVML setup
# =========================
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

# =========================
# Workload argument lists
# =========================
matmul_args = []
K = 12288
N = K
for m in range(5, 16):
    matmul_args.append((2**m, K, N, "fp16"))

softmax_args = []
M = 2**12
for n in range(5, 16):
    softmax_args.append((M, 2**n, "fp16"))
N = 2**12
for m in range(5, 16):
    softmax_args.append((2**m, N, "fp16"))

layernorm_args = []
N = 2**12
for m in range(5, 16):
    layernorm_args.append((2**m, N, "fp16"))
M = 2**12
for n in range(5, 16):
    layernorm_args.append((M, 2**n, "fp16"))

gelu_args = []
for m in range(10, 30):
    gelu_args.append((2**m, "fp16"))

# =========================
# Helpers
# =========================
def measure_idle_power(interval=0.01, idle_seconds=1.0):
    """Sample NVML briefly to estimate idle baseline (Watts)."""
    readings = []
    start = time.time()
    while time.time() - start < idle_seconds:
        try:
            p = nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            p = float("nan")
        readings.append(p)
        time.sleep(interval)
    vals = [x for x in readings if isinstance(x, (int, float))]
    return median(vals) if vals else float("nan")

def integrate_energy_joules(timestamps, powers_w):
    if len(timestamps) < 2:
        return 0.0
    e = 0.0
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]                # seconds
        p_avg = 0.5 * (powers_w[i] + powers_w[i-1])        # watts
        e += p_avg * dt                                    # joules
    return e

Total_Latency = 0.0

# =========================
# Workload runners (iterative)
# =========================
def run_for_seconds(seconds, fn):
    """Call fn() repeatedly until wall-clock 'seconds' elapse."""
    end = time.time() + seconds
    # optional warm-up to avoid first-call jitters
    avg = []
    avg.append(fn())
    while time.time() < end:
        avg.append(fn())
    global Total_Latency
    Total_Latency += sum(avg) / len(avg) if avg else 0.
    print('Latency: ', sum(avg) / len(avg) if avg else 0)

def test_matmul_iter(M, K, N, datatype, seconds=2.0):
    model = Matmul(data_type=data_type_dict[datatype])
    A = Tensor([M, K])
    B = Tensor([K, N])
    _ = model(A, B)
    return run_for_seconds(seconds, model.run_on_gpu)

def test_layernorm_iter(M, N, datatype, seconds=2.0):
    model = LayerNorm(data_type=data_type_dict[datatype])
    X = Tensor([M, N], data_type=data_type_dict[datatype])
    _ = model(X)
    return run_for_seconds(seconds, model.run_on_gpu)

def test_gelu_iter(M, datatype, seconds=2.0):
    model = GeLU(data_type=data_type_dict[datatype])
    X = Tensor([M], data_type=data_type_dict[datatype])
    _ = model(X)
    return run_for_seconds(seconds, model.run_on_gpu)

def test_softmax_iter(M, N, datatype, seconds=2.0):
    model = Softmax(data_type=data_type_dict[datatype])
    X = Tensor([M, N], data_type=data_type_dict[datatype])
    _ = model(X)
    return run_for_seconds(seconds, model.run_on_gpu)

# =========================
# Profiler
# =========================
def profile_power(
    interval, output_file, func, *args,
    idle_seconds=1.0, idle_margin_w=5.0, idle_margin_rel=0.10, **kwargs
):
    """
    Run func(*args, **kwargs) while sampling GPU power.
    Returns raw and idle-corrected metrics.
    """
    # 1) Idle baseline & threshold
    baseline_w = measure_idle_power(interval=interval, idle_seconds=idle_seconds)
    threshold_w = baseline_w + max(idle_margin_w, baseline_w * idle_margin_rel)

    power_readings, timestamps = [], []
    start_time = time.time()

    # 2) Launch workload in background
    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    thread.start()

    # 3) Sample power
    while thread.is_alive():
        try:
            p = nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            p = float("nan")
        power_readings.append(p)
        timestamps.append(time.time() - start_time)
        time.sleep(interval)

    one_time_latency = thread.join()

    end_time = time.time()
    latency = end_time - start_time  # total wall time for this run (≈ seconds + overhead)

    # 4) Plot
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if power_readings and timestamps:
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, power_readings, label=f"Power over {latency:.2f}s")
        plt.xlabel("Time (s)"); plt.ylabel("Power (W)")
        plt.title(f"Power Consumption Profile - {func.__name__}")
        plt.legend(); plt.grid()
        plt.savefig(out_path); plt.close()

    # 5) Metrics
    valid = [p for p in power_readings if isinstance(p, (int, float))]
    avg_power_total = (sum(valid)/len(valid)) if valid else float("nan")
    if valid:
        max_power = max(valid)
        peak_idx = power_readings.index(max_power)
        peak_time = timestamps[peak_idx] if 0 <= peak_idx < len(timestamps) else float("nan")
    else:
        max_power, peak_time = float("nan"), float("nan")

    # Energies (total & dynamic)
    energy_total_j = integrate_energy_joules(timestamps, power_readings) if power_readings else 0.0
    dyn_samples = [max(0.0, p - baseline_w) for p in power_readings]
    energy_dynamic_j = integrate_energy_joules(timestamps, dyn_samples) if power_readings else 0.0

    # Active vs idle-ish classification
    active_mask = [(p > threshold_w) for p in power_readings]
    active_vals = [p for p, a in zip(power_readings, active_mask) if a]
    avg_power_active = (sum(active_vals)/len(active_vals)) if active_vals else avg_power_total
    avg_dynamic_power = ((sum(v - baseline_w for v in active_vals)/len(active_vals))
                         if active_vals else float("nan"))
    active_duty_cycle = (sum(active_mask)/len(active_mask)) if active_mask else 0.0

    return {
        "latency_s": one_time_latency,
        "baseline_w": baseline_w,
        "threshold_w": threshold_w,
        "avg_power_total_w": avg_power_total,
        "avg_power_active_w": avg_power_active,
        "avg_dynamic_power_w": avg_dynamic_power,
        "max_power_w": max_power,
        "peak_time_s": peak_time,
        "energy_total_j": energy_total_j,
        "energy_dynamic_j": energy_dynamic_j,
        "energy_total_wh": energy_total_j / 3600.0,
        "energy_dynamic_wh": energy_dynamic_j / 3600.0,
        "active_duty_cycle": active_duty_cycle,
        "samples": len(power_readings),
        "output_file": str(out_path),
    }

# =========================
# Runner + summary
# =========================
def _slug(arg_tuple):
    return "_".join(str(x).replace("/", "_") for x in arg_tuple)

def run_profiles(model_name, test_func, args, output_dir, interval, seconds,
                 idle_seconds, idle_margin_w, idle_margin_rel):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    records = []
    for i, arg in enumerate(args):
        print(f"Running {model_name} test {i+1}/{len(args)} with args: {arg}")
        output_file = f"{output_dir}/{model_name}_{_slug(arg)}.png"
        metrics = profile_power(
            interval, output_file, test_func, *arg,
            seconds=seconds,
            idle_seconds=idle_seconds,
            idle_margin_w=idle_margin_w,
            idle_margin_rel=idle_margin_rel
        )
        record = {"model": model_name, "args": arg, **metrics}
        records.append(record)
        print(f"Finished {model_name} test {i+1}/{len(args)} with args: {arg}")
    return records

def write_summary(records, path="power_summary.txt", seconds=2.0, interval=0.01):
    lines = []
    lines.append("# GPU Power Profiling Summary (idle-aware, fixed-duration)\n")
    lines.append(f"- target seconds per run: {seconds}\n- sampling interval: {interval} s\n")
    by_model = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)

    for model, recs in by_model.items():
        lines.append(f"\n## {model}\n")
        for r in recs:
            lines.append(
                f"- args={r['args']}\n"
                f"  - Latency: {r['latency_s']:.4f} s\n"
                f"  - baseline (idle): {r['baseline_w']:.2f} W | threshold: {r['threshold_w']:.2f} W\n"
                f"  - avg power (total): {r['avg_power_total_w']:.2f} W\n"
                f"  - avg power (active-only): {r['avg_power_active_w']:.2f} W\n"
                f"  - avg dynamic power (active-only, minus idle): {r['avg_dynamic_power_w']:.2f} W\n"
                f"  - peak power: {r['max_power_w']:.2f} W @ {r['peak_time_s']:.3f} s\n"
                f"  - energy (total): {r['energy_total_j']:.2f} J ({r['energy_total_wh']:.6f} Wh)\n"
                f"  - energy (dynamic, >idle): {r['energy_dynamic_j']:.2f} J ({r['energy_dynamic_wh']:.6f} Wh)\n"
                f"  - active duty cycle: {100.0 * r['active_duty_cycle']:.1f}%\n"
                f"  - samples: {r['samples']}\n"
                f"  - plot: {r['output_file']}\n"
            )
    Path(path).write_text("\n".join(lines))
    print(f"\nWrote summary to {path}")

# =========================
# Main / CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Idle-aware GPU power profiler with fixed-duration runs.")
    ap.add_argument("--seconds", type=float, default=2.0, help="Wall-clock seconds per run (per case).")
    ap.add_argument("--interval", type=float, default=0.01, help="Sampling interval in seconds.")
    ap.add_argument("--idle-seconds", type=float, default=1.0, help="Seconds to measure idle baseline before each run.")
    ap.add_argument("--idle-margin-w", type=float, default=5.0, help="Absolute watts above idle to count as active.")
    ap.add_argument("--idle-margin-rel", type=float, default=0.10, help="Relative margin (fraction of idle) for active threshold.")
    ap.add_argument("--summary", type=str, default="power_summary.txt", help="Summary output path.")
    args = ap.parse_args()

    all_records = []
    all_records += run_profiles("matmul",    test_matmul_iter,    matmul_args,    "matmul_profiles",
                                args.interval, args.seconds, args.idle_seconds, args.idle_margin_w, args.idle_margin_rel)
    all_records += run_profiles("layernorm", test_layernorm_iter, layernorm_args, "layernorm_profiles",
                                args.interval, args.seconds, args.idle_seconds, args.idle_margin_w, args.idle_margin_rel)
    all_records += run_profiles("gelu",      test_gelu_iter,      gelu_args,      "gelu_profiles",
                                args.interval, args.seconds, args.idle_seconds, args.idle_margin_w, args.idle_margin_rel)
    all_records += run_profiles("softmax",   test_softmax_iter,   softmax_args,   "softmax_profiles",
                                args.interval, args.seconds, args.idle_seconds, args.idle_margin_w, args.idle_margin_rel)

    write_summary(all_records, args.summary, seconds=args.seconds, interval=args.interval)
    print("All profiling completed.")
