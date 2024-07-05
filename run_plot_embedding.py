import sys
import os
name = "2023-12-28_1"
suffix = "10000_2000_81"

cmd = f"nohup python -u src/latent_vis/plot_age.py {name} {suffix} > ./logs/plot_age.out &"
os.system(cmd)

cmd = f"nohup python -u src/latent_vis/plot_anatomical.py {name} {suffix} > ./logs/plot_anatomical.out &"
os.system(cmd)

cmd = f"nohup python -u src/latent_vis/plot_male.py {name} {suffix} > ./logs/plot_male.out &"
os.system(cmd)

cmd = f"nohup python -u src/latent_vis/plot_pathology.py {name} {suffix} > ./logs/plot_pathology.out &"
os.system(cmd)
