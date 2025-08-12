import os
from typing import List, Tuple
from itertools import product
import pandas as pd
from torch import float16

from kernels import test_matmul_iter, test_softmax_iter, run_pipeline, set_device

def prefill_pipelines() -> List[Tuple]:
    B = [1,2,4,8]               # Batch Sizes
    P = [256, 512, 1024, 2048]  # Lengths
    # M is the same as P
    O = [9216]                  # Original input embedding dimension
    D = [9216]                  # Output embedding dimension after projection
    H = [72]                    # Number of heads
    E = [128]                   # Per head embedding dimension
    F = [128]                   # Feature dim per head
    I = [36864]                 # 4× hidden dim, common in FFN

    prod = product(B, P, O, D, H, E, F, I)
    return [ gpt_OPT_66B_pipeline(b, p, p, o, d, h, e, f, i, PREFILL_ITERS) for b, p, o, d, h, e, f, i in prod ]

def decode_pipelines() -> List[Tuple]:
    B = [1,2,4,8]               # Batch Sizes
    P = [1]                     # Lengths
    M = [2048]                  # M is the output dimension for QKT
    O = [9216]                  # Original input embedding dimension
    D = [9216]                  # Output embedding dimension after projection
    H = [72]                    # Number of heads
    E = [128]                   # Per head embedding dimension
    F = [128]                   # Feature dim per head
    I = [36864]                 # 4× hidden dim, common in FFN

    prod = product(B, P, M, O, D, H, E, F, I)
    return [ gpt_OPT_66B_pipeline(b, p, m, o, d, h, e, f, i, DECODE_ITERS) for b, p, m, o, d, h, e, f, i in prod ]

DECODE_ITERS = [3000, 3000, 3000, 50000, 50000, 50000, 1000, 500, 500]
PREFILL_ITERS = [1000, 1000, 1000, 5000, 5000, 5000, 500, 500, 500]

def gpt_OPT_66B_pipeline(b, p, m, o, d, h, e, f, i, iters) -> List:
    phases = []
    phases.append(('Q-proj',        lambda: test_matmul_iter("Q-proj", b * p, o, d, float16, iters=iters[0])))
    phases.append(('K-proj',        lambda: test_matmul_iter("K-proj", b * p, o, d, float16, iters=iters[1])))
    phases.append(('V-proj',        lambda: test_matmul_iter("V-proj", b * p, o, d, float16, iters=iters[2])))
    phases.append(('QKT',           lambda: test_matmul_iter("QKT", h * b * p, e, m, float16, iters=iters[3])))
    phases.append(('Softmax',       lambda: test_softmax_iter("Softmax", h * b * p, m, float16, iters=iters[4])))
    phases.append(('AV',            lambda: test_matmul_iter("AV", h * b * p, m, f, float16, iters=iters[5])))
    phases.append(('Output-proj',   lambda: test_matmul_iter("Output-proj", b * p, h * f, o, float16, iters=iters[6])))
    phases.append(('FFN-1',         lambda: test_matmul_iter("FFN-1", b * p, o, i, float16, iters=iters[7])))
    phases.append(('Down-proj',     lambda: test_matmul_iter("Down-proj", b * p, i, o, float16, iters=iters[8])))
    name = f"B{b}_P{p}_M{m}"
    return name, phases

def save_csv_wide(all_results, filename="summary.csv"):
    # Flatten results -> long table
    df_layers = pd.json_normalize(
        all_results,
        record_path="results",
        meta=["name", "total_pipeline_latency", "total_pipeline_energy"],
        record_prefix="layer_",
        meta_prefix="pipeline_",
    )

    # Pivot to wide: metric__layer columns
    wide = (df_layers.pivot_table(
                index="pipeline_name",
                columns="layer_name",
                values=["layer_avg_latency_ms", "layer_avg_energy_J", "layer_avg_power_W"],
                aggfunc="first")
            .sort_index(axis=1, level=1))

    # Flatten MultiIndex columns
    wide.columns = [f"{metric}__{layer}" for metric, layer in wide.columns]
    wide = wide.reset_index().rename(columns={"pipeline_name": "pipeline"})

    # Add pipeline totals
    totals = (df_layers[["pipeline_name", "pipeline_total_pipeline_latency", "pipeline_total_pipeline_energy"]]
              .drop_duplicates()
              .rename(columns={
                  "pipeline_total_pipeline_latency": "pipeline_latency_ms",
                  "pipeline_total_pipeline_energy": "pipeline_energy_J",
              }))
    wide = wide.merge(totals, left_on="pipeline", right_on="pipeline_name", how="left") \
               .drop(columns=["pipeline_name"])

    num_cols = [c for c in wide.columns
            if c.startswith(("layer_avg_latency_ms__", "layer_avg_energy_J__", "layer_avg_power_W__"))
            or c in ("pipeline_latency_ms", "pipeline_energy_J")]

    wide[num_cols] = wide[num_cols].apply(pd.to_numeric, errors="coerce")
    wide[num_cols] = wide[num_cols].fillna(0.0)

    # Save
    wide.to_csv(filename, index=False)
    return wide

def pipeline_benchmark(output_dir:str, pipelines:List[Tuple], device_index:int = 0) -> List[dict]:
    set_device(device_index)

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for pipeline in pipelines:
        name, phases = pipeline
        all_results.append(run_pipeline(name, phases, output_dir, device_index))

    csv_file = os.path.join(output_dir, "summary.csv")
    save_csv_wide(all_results, filename=csv_file)
    print(f"Result summary saved to {csv_file}")

    return all_results

pipeline_benchmark(output_dir="decode_results", pipelines=decode_pipelines(), device_index=0)
pipeline_benchmark(output_dir="prefill_results", pipelines=prefill_pipelines(), device_index=0)
