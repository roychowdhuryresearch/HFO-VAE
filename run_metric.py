import os
import glob
suffix = "2023-12-28_1"
sampling = "GMM"
epochs = [int(e.split("_")[-1].split(".")[0]) for e in glob.glob(f"res/{suffix}/fold_1/ckpt/*.pth")
          if e.split("_")[-1].split(".")[0] != "best"]
epochs = [81]
for epoch in epochs:
    for n_sample_artifact in [10000, 5000]:
        for n_sample_pathological in [2000, 1000]:
            save_suffix = f"{n_sample_artifact}_{n_sample_pathological}_{epoch}"
            cmd = f"python cluster.py {suffix} {save_suffix} {epoch}"
            os.system(cmd)
            cmd = f"python auc.py {suffix} {sampling} {save_suffix}"
            os.system(cmd)
            cmd = f"python auc_kfold.py {suffix} {sampling} {save_suffix}"
            os.system(cmd)
            cmd = f"python src/fit_embedding.py {suffix} {save_suffix}"
            os.system(cmd)
            cmd = f"python random_forest.py {suffix} {save_suffix}"
            os.system(cmd)
            cmd = f"python draw/calculate_rate.py {suffix} {save_suffix}"
