import torch
import time
import os
import shutil
from typing import List, Tuple, Callable

from pynvml import nvmlDeviceGetTotalEnergyConsumption, nvmlInit, nvmlDeviceGetHandleByIndex


COLS = shutil.get_terminal_size().columns
GREEN_DOT = "\033[32m.\033[0m"

handle = None  # Global variable to hold NVML handle

def set_device(device_index: int):
    """
    Initialize NVML and set the global handle for energy consumption measurement.
    
    Args:
        device_index (int): Index of the GPU device.
    """
    global handle
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_index)
    torch.cuda.set_device(device_index)
    print(f"Using CUDA device: {torch.cuda.get_device_name(device_index)}")

UNROLL_FACTOR = 100

def test_kernel_iter(name: str, setup_func: Callable, capture_func: Callable, iters: int = 100):

    global handle
    state = setup_func()

    # Warmup
    for _ in range(10):
        capture_func(state)
    torch.cuda.synchronize()

    # CUDA Graph Capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(UNROLL_FACTOR):
            capture_func(state)

    # Timing setup
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    e0_mJ = nvmlDeviceGetTotalEnergyConsumption(handle) # NVML energy before (mJ)
    t0 = time.perf_counter()  # wall clock start

    # Run graph
    start_evt.record()
    for _ in range(iters//UNROLL_FACTOR):
        g.replay()
    end_evt.record()
    torch.cuda.synchronize()

    iters = UNROLL_FACTOR * (iters // UNROLL_FACTOR)  # Adjust iters to be a multiple of UNROLL_FACTOR

    t1 = time.perf_counter()  # wall clock end
    e1_mJ = nvmlDeviceGetTotalEnergyConsumption(handle)
    total_energy_J = (e1_mJ - e0_mJ) / 1000.0
    avg_energy_J = total_energy_J / iters
    avg_power_W = total_energy_J / (t1 - t0)

    total_latency_ms = start_evt.elapsed_time(end_evt)  # in milliseconds
    avg_latency_ms = total_latency_ms / iters 

    return {
        'name': name,
        'iters': iters,
        'avg_latency_ms': avg_latency_ms,
        'avg_energy_J': avg_energy_J,
        'avg_power_W': avg_power_W
    }

def test_matmul_iter(name: str, N:int, M:int, K:int, datatype:torch.dtype=torch.float16, iters:int=100):
    def setup():
        input1 = torch.randn(N, M, dtype=datatype, device='cuda')
        input2 = torch.randn(M, K, dtype=datatype, device='cuda')
        output = torch.empty(N, K, dtype=datatype, device='cuda')
        return input1, input2, output

    def capture(state):
        input1, input2, output = state
        torch.matmul(input1, input2, out=output)
        return output

    return test_kernel_iter(name, setup, capture, iters)

def test_softmax_iter(name: str,N:int, M:int, datatype:torch.dtype=torch.float16, iters:int=100):
    def setup():
        input_tensor = torch.randn(N, M, dtype=datatype, device='cuda')
        output_tensor = torch.empty_like(input_tensor)
        return input_tensor, output_tensor

    def capture(state):
        input_tensor, output_tensor = state
        torch.softmax(input_tensor, dim=-1, out=output_tensor)
        return output_tensor

    return test_kernel_iter(name, setup, capture, iters)

# def test_conv_iter(name: str, N:int, C:int, H:int, W:int, K:int, datatype:torch.dtype=torch.float16, iters:int=100):
#     def setup():
#         input_tensor = torch.randn(N, C, H, W, dtype=datatype, device='cuda')
#         weight = torch.randn(K, C, 3, 3, dtype=datatype, device='cuda')
#         output_tensor = torch.empty(N, K, H - 2, W - 2, dtype=datatype, device='cuda')
#         return input_tensor, weight, output_tensor

#     def capture(state):
#         input_tensor, weight, output_tensor = state
#         torch.nn.functional.conv2d(input_tensor, weight, out=output_tensor)
#         return output_tensor

#     return test_kernel_iter(name, setup, capture, iters)


def run_phase(res_list: List[dict], func: Callable, *args, **kwargs):
    res_list.append(func(*args, **kwargs))
    print(GREEN_DOT, end='', flush=True)

def run_pipeline(pipe_name, phase_list:List[Tuple[str, Callable]], output_dir, device_index: int = 0) -> dict:
    results = []

    start_time = time.perf_counter()

    print(f"[ Running Pipeline {pipe_name} ]", end=' ')
    for _, lambda_func in phase_list:
        run_phase(results, lambda_func)

    end_time = time.perf_counter()
    print(" " * (60 - len(pipe_name)) + "  [ {:>6.2f}s ]".format(end_time - start_time))

    total_pipeline_latency = sum(result['avg_latency_ms'] for result in results)
    total_pipeline_energy = sum(result['avg_energy_J'] for result in results)

    output_file = os.path.join(output_dir, f"{pipe_name}.txt")
    with open(output_file, 'w') as f:
        f.write(f"Results for {pipe_name}:\n")
        for result in results:
            f.write(f"{result}\n")
        f.write(f"Total Pipeline Latency (ms): {total_pipeline_latency}\n")
        f.write(f"Total Pipeline Energy (J): {total_pipeline_energy}\n")

    return {
        'name': pipe_name,
        'results': results,
        'total_pipeline_latency': total_pipeline_latency,
        'total_pipeline_energy': total_pipeline_energy,
        'layer_results': results
    }