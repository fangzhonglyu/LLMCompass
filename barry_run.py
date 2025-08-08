from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage
import time, threading
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU
from software_model.matmul import Matmul
from software_model.softmax import Softmax
from software_model.utils import data_type_dict, Tensor
from matplotlib import pyplot as plt
import argparse
import os

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

# matmul args
matmul_args = []
K = 12288
N = K
for m in range(5, 16):
    matmul_args.append((2**m, K, N, "fp16"))

# Softmax args
softmax_args = []
M = 2**12
for n in range(5, 16):
    softmax_args.append((M, 2**n, "fp16"))
N = 2**12
for m in range(5, 16):
    softmax_args.append((2**m, N, "fp16"))

# LayerNorm args
layernorm_args = []
N = 2**12
for m in range(5, 16):
    layernorm_args.append((2**m, N, "fp16"))
M = 2**12
for n in range(5, 16):
    layernorm_args.append((M, 2**n, "fp16"))

# GeLU args
gelu_args = []
for m in range(10, 30):
    gelu_args.append((2**m, "fp16"))

def test_matmul(M, K, N, datatype, test_overhead=False):
    model = Matmul(data_type=data_type_dict[datatype])
    _ = model(
        Tensor([M, K]),
        Tensor([K, N]),
    )
    if test_overhead:
        model.gpu_kernel_launch_overhead()
        test_overhead = False
    latency = model.run_on_gpu()
    return latency

def test_layernorm(M, N, datatype):
    model = LayerNorm(data_type=data_type_dict[datatype])
    _ = model(Tensor([M, N], data_type=data_type_dict[datatype]))
    latency = model.run_on_gpu()
    return latency

def test_gelu(M, datatype):
    model = GeLU(data_type=data_type_dict[datatype])
    _ = model(Tensor([M], data_type=data_type_dict[datatype]))
    latency = model.run_on_gpu()
    return latency

def test_softmax(M, N, datatype):
    model = Softmax(data_type=data_type_dict[datatype])
    _ = model(Tensor([M, N], data_type=data_type_dict[datatype]))
    latency = model.run_on_gpu()
    return latency


# Profile GPU power consumption of a function, collecting data every interval seconds
# and plotting the results.
def profile_power(interval, output_file, func, *args, **kwargs):
    power_readings = []
    timestamps = []
    start_time = time.time()
    
    # launch function in a separate thread
    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    thread.start()

    while thread.is_alive():
        power = nvmlDeviceGetPowerUsage(handle) / 1000.0
        power_readings.append(power)
        timestamps.append(time.time() - start_time)
        time.sleep(interval)
    
    end_time = time.time()
    latency = end_time - start_time
    thread.join()

    # Plot power consumption over time
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, power_readings, label=f"Power Graph Over '{latency:.2f}s'", color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title(f"Power Consumption Profile - {func.__name__}")
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    plt.close()

    return latency

def run_profiles(model_name, test_func, args, output_dir):
    for i, arg in enumerate(args):
        print(f"Running {model_name} test {i+1}/{len(args)} with args: {arg}")
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{model_name}_{arg}.png"
        profile_power(0.01, output_file, test_func, *arg)
        print(f"Finished {model_name} test {i+1}/{len(args)} with args: {arg}")


run_profiles("matmul", test_matmul, matmul_args, "matmul_profiles")
run_profiles("layernorm", test_layernorm, layernorm_args, "layernorm_profiles")
run_profiles("gelu", test_gelu, gelu_args, "gelu_profiles")
run_profiles("softmax", test_softmax, softmax_args, "softmax_profiles")
print("All profiling completed.")
