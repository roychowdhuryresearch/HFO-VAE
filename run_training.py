PYTORCH_NO_CUDA_MEMORY_CACHING=0
import torch
import time
import os
import subprocess
import os
import datetime
import time
import glob
PYTORCH_NO_CUDA_MEMORY_CACHING=0
import subprocess

def get_gpu_info(A5000=True,Titan=True,RTX2080=True):
    """Get information about available GPUs and their memory usage"""
    gpu_output = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv'])
    gpu_lines = gpu_output.decode().split('\n')
    gpu_summary = []
    gpus = []
    assert (A5000 or Titan or RTX2080), "No GPUs specified."
    for line in gpu_lines[1:]:
        if line.strip() != '':
            gpu_name, gpu_memory_total, gpu_memory_used, gpu_memory_free = line.split(',')
            if "A5000" in gpu_name and A5000:
                gpu_summary.append(int(gpu_memory_used.split()[0]))
                gpus.append(gpu_name)
            elif "TITAN" in gpu_name and Titan:
                gpu_summary.append(int(gpu_memory_used.split()[0]))
                gpus.append(gpu_name)
            elif "2080" in gpu_name and RTX2080:
                gpu_summary.append(int(gpu_memory_used.split()[0]))
                gpus.append(gpu_name)
            else:
                print(f"GPU {gpu_name} is not being used.")
    return gpus, gpu_summary

def get_gpu_info_pytorch():
    """Get information about available GPUs and their memory usage"""
    gpu_summary = []
    gpus = []
    for i in range(torch.cuda.device_count()):
        # if i in GPUs_to_exclude:
        #     continue
        gpu_name = torch.cuda.get_device_name(i)
        f,t = torch.cuda.mem_get_info(i)
        gpu_memory_used = int((t-f)/1024/1024)
        gpu_summary.append(gpu_memory_used)
        gpus.append(gpu_name)
    return gpus, gpu_summary

if __name__ == "__main__":
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir = f"logs/{timestamp}"
    # logs_dir = "logs/2023-10-16_18-42-39"
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    run_n = glob.glob("./res"+f"/{today}*")
    print("there are already",len(run_n),"runs today")
    today = today + f"_{len(run_n)}"
    os.makedirs(logs_dir, exist_ok=True)
    torch.cuda.init()

    start = time.time()
    fold_num = 0
    expected_fold = 5
    exclude_list = [3,4,5]
    max_gpus_used = 3
    n_gpus_used = 0
    gpus_used = []
    while fold_num < expected_fold:
        gpu_names, gpu_memory = get_gpu_info_pytorch()
        print(f"GPUs: {gpu_names}")
        print(f"GPU memory: {gpu_memory}")
        # raise Exception
        for i, memory_used in enumerate(gpu_memory):
            if memory_used < 1600:
                print(f"GPU {gpu_names[i]} has {memory_used} MB of memory used.")
                if i in gpus_used:
                    n_gpus_used -= 1
                    gpus_used.remove(i)
                if fold_num == expected_fold:
                    break
                if i not in exclude_list and n_gpus_used < max_gpus_used:
                    cmd = f"CUBLAS_WORKSPACE_CONFIG=:16:8 nohup python -u src/trainer.py {fold_num} cuda:{i} {today} {expected_fold} train > {logs_dir}/fold_{fold_num}.log &"
                    os.system(cmd)
                    fold_num += 1
                    print(f"running fold {fold_num-1} on GPU {gpu_names[i]}")
                    n_gpus_used += 1
                    gpus_used.append(i)
                else: 
                    print(f"cuda{i}: GPU {gpu_names[i]} has less than 1GB of memory available, but no code has been specified to run on that GPU.")
            else:
                print(f"GPU {gpu_names[i]} has {memory_used} MB of memory used. Skipping.")
        print(f"using {n_gpus_used} GPUs, gpus used are:",end = "")
        for gpu in gpus_used:
            print(f" cuda:{gpu} ({gpu_names[gpu]})", end = "")
        print()
        print("sleeping for 240 seconds")
        time.sleep(240*8) # Check every 60 seconds
    end = time.time()
    # total time taken in minutes
    print(f"Total time: {(end - start)/60} minutes")
