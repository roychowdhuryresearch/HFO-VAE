import sys
import os

name = "2023-12-28_1"
suffix = "10000_2000_81"

cmd = f"nohup python -u src/characteristics/ttest_all.py {name} {suffix} > ./logs/ttest_all.out &"
os.system(cmd)

cmd = f"nohup python -u src/characteristics/plot_embedding.py {name} {suffix} > logs/plot_embedding.out &"
os.system(cmd)

ttest_type = "patient"
cmd = f"nohup python -u src/characteristics/ttest_age.py {ttest_type} {name} {suffix} > logs/ttest_age.out &"
os.system(cmd)

cmd = f"nohup python -u src/characteristics/ttest_dataset.py {ttest_type} {name} {suffix} > ./logs/ttest_dataset.out &"
os.system(cmd)

cmd = f"nohup python -u src/characteristics/ttest_gender.py {ttest_type} {name} {suffix} > ./logs/ttest_gender.out &"
os.system(cmd)

cmd = f"nohup python -u src/characteristics/ttest_anatomical.py {ttest_type} {name} {suffix} > ./logs/ttest_anatomical.out &"
os.system(cmd)

cmd = f"nohup python -u src/characteristics/ttest_pathology.py {ttest_type} {name} {suffix} > ./logs/ttest_pathology.out &"
os.system(cmd)
