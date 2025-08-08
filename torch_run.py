import math, time, torch
from gpt_run import profile_power

# ---- config ---------------------------------------------------
device = "cuda"
dtype  = torch.float16

def benchmark_attn_prefill(b, p, o, h, e, i, seconds=2.0):
    """
    Prefill attention + FFN(out proj only) with per-phase CUDA-event timings.
    Assumes: o == h * e  (hidden size = heads * head_dim)
    Returns: dict of per-phase latencies (s) and total (s).
    """
    assert o == h * e, "o must equal h * e (hidden size)."
    d_model = o
    D = d_model  # alias for clarity
    F = e        # per-head dim
    M = p        # key len == query len in prefill

    # ---- weights (random, fixed) ----
    Wq = torch.empty((D, D), device=device, dtype=dtype).normal_(0, 0.02)
    Wk = torch.empty((D, D), device=device, dtype=dtype).normal_(0, 0.02)
    Wv = torch.empty((D, D), device=device, dtype=dtype).normal_(0, 0.02)
    Wo = torch.empty((D, D), device=device, dtype=dtype).normal_(0, 0.02)

    # ---- activations ----
    X = torch.randn((b, p, D), device=device, dtype=dtype)   # input tokens

    stream = torch.cuda.current_stream()
    evt = lambda: torch.cuda.Event(enable_timing=True)  # ms timing on GPU clock

    # warmup a couple times to stabilize clocks
    for _ in range(3):
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        Q = Q.view(b, p, h, F).transpose(1, 2)          # (b,h,p,F)
        K = K.view(b, p, h, F).transpose(1, 2)          # (b,h,p,F)
        V = V.view(b, p, h, F).transpose(1, 2)          # (b,h,p,F)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(F)  # (b,h,p,m)
        P_attn = torch.softmax(scores, dim=-1)                          # (b,h,p,m)
        ctx = torch.matmul(P_attn, V)                                    # (b,h,p,F)
        Y = ctx.transpose(1, 2).contiguous().view(b, p, D)               # (b,p,D)
        O  = Y @ Wo
    torch.cuda.synchronize()

    # ---- timed loop: record one full pass repeatedly for ~seconds ----
    t_end = time.time() + seconds
    totals = {k: 0.0 for k in ["Q_proj","K_proj","V_proj","QK","Softmax","AV","O_proj","TOTAL"]}
    iters = 0

    while time.time() < t_end:
        # fresh input each iter if you want (optional)
        # X = torch.randn((b, p, D), device=device, dtype=dtype)

        # events
        e0 = evt(); e1 = evt(); e2 = evt(); e3 = evt(); e4 = evt(); e5 = evt(); e6 = evt(); e7 = evt(); e8 = evt()
        # start
        e0.record(stream)

        # Q
        Q = X @ Wq
        e1.record(stream)   # end Q
        # K
        K = X @ Wk
        e2.record(stream)   # end K
        # V
        V = X @ Wv
        e3.record(stream)   # end V

        # reshape to (b,h,p,F)
        Q = Q.view(b, p, h, F).transpose(1, 2)
        K = K.view(b, p, h, F).transpose(1, 2)
        V = V.view(b, p, h, F).transpose(1, 2)

        # QK^T
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(F)   # (b,h,p,m)
        e4.record(stream)   # end QK

        # Softmax over m
        P_attn = torch.softmax(scores, dim=-1)
        e5.record(stream)   # end softmax

        # A * V
        ctx = torch.matmul(P_attn, V)                                  # (b,h,p,F)
        e6.record(stream)   # end AV

        # concat heads -> (b,p,D)
        Y = ctx.transpose(1, 2).contiguous().view(b, p, D)

        # output projection
        O = Y @ Wo
        e7.record(stream)   # end O_proj

        # end total
        e8.record(stream)

        torch.cuda.synchronize()

        # accumulate times (convert ms→s)
        totals["Q_proj"]  += e0.elapsed_time(e1) * 1e-3
        totals["K_proj"]  += e1.elapsed_time(e2) * 1e-3
        totals["V_proj"]  += e2.elapsed_time(e3) * 1e-3
        totals["QK"]      += e3.elapsed_time(e4) * 1e-3
        totals["Softmax"] += e4.elapsed_time(e5) * 1e-3
        totals["AV"]      += e5.elapsed_time(e6) * 1e-3
        totals["O_proj"]  += e6.elapsed_time(e7) * 1e-3
        totals["TOTAL"]   += e0.elapsed_time(e8) * 1e-3
        iters += 1

    # average per full pass
    for k in totals:
        totals[k] /= max(1, iters)
    return {"iters": iters, **totals}


if __name__ == "__main__":
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = True  # like many prod settings
    torch.backends.cudnn.allow_tf32 = True

    # Example: your 72×128 heads; P=2048 prefill
    b, p, h, e = 1, 2048, 72, 128
    o = h * e
    i = 4 * o

    profile_power(0.01, "GPT-OPT 66B Prefill", benchmark_attn_prefill, b, p, o, h, e, i, seconds=2.0)

    # stats = benchmark_attn_prefill(b, p, o, h, e, i, seconds=2.0)
    # print(stats)
