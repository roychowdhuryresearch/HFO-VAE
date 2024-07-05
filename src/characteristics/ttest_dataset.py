
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import sys
# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_embedding import create_df
from ttest_helper import ttest_patients, t_test_all_samples

def ttest_all(df, col, savefn):
    t_test_ = []
    datasets = ("UCLA Grid&Strip","SEEG","Detroit Grid&Strip")
    #print the df["anatomy"].unique() and counts
    fig, axs = plt.subplots(1,len(datasets), figsize=(3*len(datasets), 4.5), sharey=True)
    for i in range(len(datasets)):
        if datasets[i] == "Detroit Grid&Strip":
            df_ = df[df["pt_names"].str.startswith("M")]
        elif datasets[i] == "SEEG":
            df_ = df[df["pt_names"].str.startswith("SEEG")]
        else:
            df_ = df[df["pt_names"].str.startswith("Pt")]
        if t_test_type == "patient":
            t_test_result = ttest_patients(df_, col)
            t_test_result = np.stack(t_test_result)
            mean_t_test = np.mean(t_test_result, axis=0)
        else:
            t_test_result = t_test_all_samples(df_, col)
            mean_t_test = t_test_result[0]
        try:
            axs[i].imshow(mean_t_test)
        except:
            print(datasets[i], len(t_test_result))
        t_test_.append(mean_t_test)
        axs[i].set_title(f"{datasets[i]} {df_['pt_names'].unique().shape[0]} subjects")
        # yaixticks = 10-290 hz 
        # xaxis ticks = -285 - 285 ms
        # image shape = 64 * 64 
        axs[i].set_xticks(np.linspace(0, 63, 5))
        axs[i].set_xticklabels(np.linspace(-285, 285, 5).astype(int))
        axs[i].set_yticks(np.linspace(0, 63, 5))
        axs[i].set_yticklabels(np.linspace(10, 290, 5).astype(int)[::-1])
        axs[i].set_xlabel("Time (ms)", fontsize=15)
        if i==0:
            axs[i].set_ylabel("Frequency (Hz)", fontsize=15)
    np.savez(savefn.replace(".png", ".npz"), t_test=t_test_, col=datasets)
    # super title
    plt.suptitle("T-test of pathological vs physiological \n in different datasets", fontsize=20)
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()


if __name__ == "__main__":
    t_test_type = sys.argv[1]
    name = sys.argv[2]
    suffix = sys.argv[3]
    paths = glob.glob(f"./res/{name}/fold_*/{suffix}")
    embedding = "train_embedding.csv"
    label_path = "train_overall_.npz"
    df_all = []
    for path in paths:
        df_test = create_df(path,label_path.replace("train","test"),embedding.replace("train","test"))
        df_all.append(df_test)
    df_all = pd.concat(df_all)
    df_all = df_all[df_all["vae-predict"]!="Artifact"]
    df_all["pred"] = df_all["vae-predict"].replace({"Physiological":0,"Pathological":1})
    save_fn = os.path.join(f"./res/{name}/ttest" ,"ttest_dataset.png")
    os.makedirs(os.path.dirname(save_fn), exist_ok=True)
    ttest_all(df_all, "pred", save_fn)
    
