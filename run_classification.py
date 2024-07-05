import os
import time
import glob
suffix = "2023-12-28_1"
gpu_list = [1,2,3]
os.makedirs(f"logs/temp", exist_ok=True)
epochs = [e.split("_")[-1].split(".")[0] for e in glob.glob(f"res/{suffix}/fold_1/ckpt/*.pth") if e.split("_")[-1].split(".")[0] != "best"]
epochs = [81]
for epoch in epochs:
    print("working on epoch",epoch,"...")
    logs = []
    for fold_num in range(5):
        print("here")
        cmd = f"CUBLAS_WORKSPACE_CONFIG=:16:8 python src/trainer.py {fold_num} cuda:{gpu_list[fold_num%len(gpu_list)]} {suffix} {5} test {epoch} > logs/temp/{fold_num}.log 2>&1 &"
        logs.append(f"logs/temp/{fold_num}.log")
        os.system(cmd)
    time.sleep(60*10)
