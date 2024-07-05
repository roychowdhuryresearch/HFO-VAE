import sys
import os

name = "2023-12-28_1"
suffix = "10000_2000_81"
cmd = f"python src/metric/ablation_f1_all.py {name}"
os.system(cmd)
cmd = f"python src/metric/ablation_auc_all.py {name}"
os.system(cmd)
cmd = f"python draw/calculate_rate.py {name} {suffix}"
os.system(cmd)
cmd = f"python draw/figure2.py {name} {suffix}"
os.system(cmd)
