
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
    t_test_result = ttest_patients(df, col)
    for rs in t_test_result:
        if type(rs) == tuple:
            print(rs)
    t_test_result = np.stack(t_test_result)
    mean_t_test = np.mean(t_test_result, axis=0)
    # x axis +- 285 ms 
    # y axis 10-290 hz 
    # mean_t_test is 64 * 64
    plt.figure(figsize=(5,5))
    plt.xticks(np.linspace(0, 63, 5), np.linspace(-285, 285, 5).astype(int))
    plt.yticks(np.linspace(0, 63, 5), np.linspace(290, 10, 5).astype(int))
    plt.xlabel("Time (ms)", fontsize= 15)
    plt.ylabel("Frequency (Hz)", fontsize= 15)
    plt.tight_layout()
    plt.imshow(mean_t_test)
    #plt.colorbar()
    plt.savefig(savefn, dpi=300)
    np.save("t_test.npy", t_test_result)


if __name__ == "__main__":
    import sys
    date = sys.argv[1]
    suffix = sys.argv[2]
    folder = f"./res/{date}/"

    save_fn = os.path.join(folder, "ttest" ,"ttest_all.png")
    os.makedirs(os.path.dirname(save_fn), exist_ok=True)
    paths = glob.glob(f'{folder}/fold_*/{suffix}')
    embedding = "train_embedding.csv"
    label_path = "train_overall_.npz"
    df_all = []
    for path in paths:
        df_test = create_df(path,label_path.replace("train","test"),embedding.replace("train","test"))
        df_all.append(df_test)
    df_all = pd.concat(df_all)
    print("before", df_all.shape)
    df_all = df_all[df_all["vae-predict"]!="Artifact"]
    print("after", df_all.shape)
    df_all["pred"] = df_all["vae-predict"].replace({"Physiological":0,"Pathological":1})
    ttest_all(df_all, "pred", save_fn)
