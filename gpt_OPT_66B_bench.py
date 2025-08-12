import os
from typing import List, Tuple
from itertools import product
import pandas as pd
from torch import float16

from kernels import test_matmul_iter, test_softmax_iter, run_pipeline, set_device

def prefill_params() -> List[Tuple]:
    B = [1,2,4,8]               # Batch Sizes
    P = [256, 512, 1024, 2048]  # Lengths
    M = [1, 2048]               # M is the output dimension for QKT
    O = [9216]                  # Original input embedding dimension
    D = [9216]                  # Output embedding dimension after projection
    H = [72]                    # Number of heads
    E = [128]                   # Per head embedding dimension
    F = [128]                   # Feature dim per head
    I = [36864]                 # 4× hidden dim, common in FFN
    return product(B, P, M, O, D, H, E, F, I)

def decode_params() -> List[Tuple]:
    B = [1,2,4,8]               # Batch Sizes
    P = [1]                     # Lengths
    M = [2048]                  # M is the output dimension for QKT
    O = [9216]                  # Original input embedding dimension
    D = [9216]                  # Output embedding dimension after projection
    H = [72]                    # Number of heads
    E = [128]                   # Per head embedding dimension
    F = [128]                   # Feature dim per head
    I = [36864]                 # 4× hidden dim, common in FFN
    return product(B, P, M, O, D, H, E, F, I)

def gpt_OPT_66B_pipeline(b, p, m, o, d, h, e, f, i) -> List:
    phases = []
    phases.append(('Q-proj',        lambda: test_matmul_iter("Q-proj", b * p, o, d, float16, iters=100)))
    phases.append(('K-proj',        lambda: test_matmul_iter("K-proj", b * p, o, d, float16, iters=100)))
    phases.append(('V-proj',        lambda: test_matmul_iter("V-proj", b * p, o, d, float16, iters=100)))
    phases.append(('QKT',           lambda: test_matmul_iter("QKT", h * b * p, e, m, float16, iters=100)))
    phases.append(('Softmax',       lambda: test_softmax_iter("Softmax", h * b * p, m, float16, iters=100)))
    phases.append(('AV',            lambda: test_matmul_iter("AV", h * b * p, m, f, float16, iters=100)))
    phases.append(('Output-proj',   lambda: test_matmul_iter("Output-proj", b * p, h * f, o, float16, iters=100)))
    phases.append(('FFN-1',         lambda: test_matmul_iter("FFN-1", b * p, o, i, float16, iters=100)))
    phases.append(('Down-proj',     lambda: test_matmul_iter("Down-proj", b * p, i, o, float16, iters=100)))
    name = f"B{b}_P{p}_M{m}_O{o}_D{d}_H{h}_E{e}_F{f}_I{i}"
    return name, phases

def pipeline_benchmark(output_dir:str, param_list:List[Tuple], device_index:int = 0) -> List[dict]:
    set_device(device_index)

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for param in param_list:
        b, p, m, o, d, h, e, f, i = param
        name, phases = gpt_OPT_66B_pipeline(b, p, m, o, d, h, e, f, i)
        all_results.append(run_pipeline(name, phases, output_dir, device_index))
    df = pd.DataFrame(all_results)
    csv_file = os.path.join(output_dir, "pipeline_benchmark_results.csv")
    df.to_csv(csv_file, index=False)
    print(f"Result Summary saved to {csv_file}")

    return all_results

pipeline_benchmark(output_dir="decode_results", param_list=decode_params(), device_index=0)
pipeline_benchmark(output_dir="prefill_results", param_list=prefill_params(), device_index=0)


        