from gpt_run import test_matmul_iter, profile_power, test_softmax_iter, set_total_latency, get_total_latency
from itertools import product
import sys, os
from contextlib import redirect_stdout


# Batch Sizes
B = [1, 2, 4, 8]

# Lengths
P = [256, 512, 1024, 2048]

# Original input embedding dimension
O = [9216]

# Output embedding dimension after projection
D = [9216]

# Number of heads
H = [72]

# Per head embedding dimension
E = [128]

# Feature dim per head 
F = [128]

# 4× hidden dim, common in FFN
I = [36864]


def gpt_OPT_66B_prefill(b, p, o, d, h, e, i):
    """
    Test the GPT-OPT 66B model in prefill mode with given parameters.
    
    Args:
        b (int): Batch size.
        p (int): Sequence length.
        o (int): Original input embedding dimension.
        d (int): Output embedding dimension after projection.
        h (int): Number of attention heads.
        e (int): Per head embedding dimension.
        i (int): Feature dimension in the feed-forward network.
    """

    m = p
    f = e

    print("Testing GPT-OPT 66B Prefill with parameters:")
    print(f" B = {b}, P = {p}, O = {o}, D = {d}, H = {h}, E = {e}, I = {i}")


    # Q projection
    test_phase('Q-proj', test_matmul_iter, b*p, o, d, "fp16", seconds=2.0)
    # K projection
    test_phase('K-proj', test_matmul_iter, b*p, o, d, "fp16", seconds=2.0)
    # V projection
    test_phase('V-proj', test_matmul_iter, b*p, o, d, "fp16", seconds=2.0)

    # QKT
    test_phase('QKT', test_matmul_iter, h*b*p, e, m, "fp16", seconds=2.0)
    # Softmax
    test_phase('Softmax', test_softmax_iter, h*b*p, m, "fp16", seconds=2.0)
    # AV
    test_phase('AV', test_matmul_iter, h*b*p, m, f, "fp16", seconds=2.0)

    # Output projection
    test_phase('Output-proj', test_matmul_iter, b*p, h*f, o, "fp16", seconds=2.0)

    #FFN
    # First matmul in FFN
    test_phase('FFN-1', test_matmul_iter, b*p, o, i, "fp16", seconds=2.0)
    
    # Down projection
    test_phase('Down-proj', test_matmul_iter, b*p, i, o, "fp16", seconds=2.0)

    print("Total Latency for GPT-OPT 66B Prefill:" + str(get_total_latency()))

def test_phase(phase_name, func, *args, **kwargs):
    # record the start latency and power for the given phase

    print('Profiling phase:', phase_name)
    print(' - ', end='')
    result = profile_power(0.01, 'outputs/'+ phase_name, func, *args, **kwargs)
    # print(result)
    print(f" - Active Power: {result['avg_power_active_w']:.2f}W")


if __name__ == "__main__":
    os.makedirs("outputs/gpt_OPT_66B_prefill", exist_ok=True)
    for b, p, o, d, h, e, i in product(B, P, O, D, H, E, I):
        print(f"Running GPT-OPT 66B Prefill with B={b}, P={p}, O={o}, D={d}, H={h}, E={e}, I={i}")
        with open(f"outputs/gpt_OPT_66B_prefill/batch_{b}_seq_{p}.txt", "w") as f:
             with redirect_stdout(f):
                set_total_latency(0.0)
                gpt_OPT_66B_prefill(b, p, o, d, h, e, i)
