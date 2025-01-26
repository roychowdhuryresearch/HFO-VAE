import sys
import os

name = "2023-12-28_1_old"
suffix = "10000_2000_81"

cmd = f"nohup python -u src/perturbations/perturbation.py {name} {suffix} cuda:0 > ./logs/perturbation.out &"
os.system(cmd)

cmd = f"nohup python -u src/perturbations/perturbation1.py {name} {suffix} cuda:1 > ./logs/perturbation1.out &"
os.system(cmd)

cmd = f"nohup python -u src/perturbations/perturbation2.py {name} {suffix} cuda:2 > ./logs/perturbation2.out &"
os.system(cmd)

cmd = f"nohup python -u src/perturbations/perturbation3.py {name} {suffix} cuda:3 > ./logs/perturbation3.out &"
os.system(cmd)
