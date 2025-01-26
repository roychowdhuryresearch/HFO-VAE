"""
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, default="/mnt/SSD5/yipeng/VAE/res/2023-12-28_3/fold_0/train_81.npz",
                    help="Path to the 'train' NPZ file.")
parser.add_argument("--test_file", type=str, default="/mnt/SSD5/yipeng/VAE/res/2023-12-28_3/fold_0/test_81.npz",
                    help="Path to the 'test' NPZ file.")
parser.add_argument("--artifact_num_samples", type=int, default=10000,
                    help="Number of samples for artifact GMM training.")
parser.add_argument("--pathology_num_samples", type=int, default=3000,
                    help="Number of samples for pathologic GMM training.")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
parser.add_argument("--save_suffix", type=str, default="results",
                    help="Folder name for saving predictions.")
args = parser.parse_args()
"""
import os
train_file = "/mnt/SSD5/yipeng/VAE/res/2023-12-28_3/fold_4/train_81.npz"
test_file = "/mnt/SSD5/yipeng/VAE/res/2023-12-28_3/fold_4/test_81.npz"
cmd = f"python cluster_s.py --train_file {train_file} --test_file {test_file} --artifact_num_samples 10000 --pathology_num_samples 3000 --seed 42 --save_suffix results"
os.system(cmd)
print("-------------------")
train_file = "/mnt/SSD5/yipeng/VAE/res/2023-12-28_3/fold_0/train_81.npz"
test_file = "/mnt/SSD5/yipeng/VAE/res/2023-12-28_3/fold_0/test_81.npz"
cmd = f"python cluster_s.py --train_file {train_file} --test_file {test_file} --artifact_num_samples 10000 --pathology_num_samples 3000 --seed 42 --save_suffix results"
os.system(cmd)
print("-------------------")
train_file = "/mnt/SSD5/yipeng/VAE/res/2023-12-28_3/fold_1/train_81.npz"
test_file = "/mnt/SSD5/yipeng/VAE/res/2023-12-28_3/fold_1/test_81.npz"
cmd = f"python cluster_s.py --train_file {train_file} --test_file {test_file} --artifact_num_samples 10000 --pathology_num_samples 3000 --seed 42 --save_suffix results"
os.system(cmd)
print("-------------------")
train_file = "/mnt/SSD5/yipeng/VAE/res/2023-12-28_3/fold_2/train_81.npz"
test_file = "/mnt/SSD5/yipeng/VAE/res/2023-12-28_3/fold_2/test_81.npz"
cmd = f"python cluster_s.py --train_file {train_file} --test_file {test_file} --artifact_num_samples 10000 --pathology_num_samples 3000 --seed 42 --save_suffix results"
os.system(cmd)
print("-------------------")
train_file = "/mnt/SSD5/yipeng/VAE/res/2023-12-28_3/fold_3/train_81.npz"
test_file = "/mnt/SSD5/yipeng/VAE/res/2023-12-28_3/fold_3/test_81.npz"
cmd = f"python cluster_s.py --train_file {train_file} --test_file {test_file} --artifact_num_samples 10000 --pathology_num_samples 3000 --seed 42 --save_suffix results"
os.system(cmd)
print("-------------------")

