import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import sys
from multiprocessing import Pool
# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_embedding import create_df
from ttest_helper import ttest_patients, t_test_all_samples, load_patient, ttest
from scipy import stats


def retrive_one_patient(data_folder, pt_name, col, df, value):
    pt_df, data = load_patient(data_folder, pt_name)
    pt_df = pt_df[["pt_names", "channel_name", "start", "end", "index"]]
    df.drop(columns=["index"], inplace=True)
    pt_df = pt_df.merge(df, on=["pt_names", "channel_name", "start", "end"])
    pos_index = pt_df[pt_df[col] == value]["index"].values
    neg_index = pt_df[pt_df[col] != value]["index"].values
    pos_data = data[pos_index]
    neg_data = data[neg_index]
    print(pt_df[col].unique(), value, "pos", pos_data.shape, "neg", neg_data.shape)
    return pos_data, neg_data

def retrive_patients(df, col, value):
    pt_names = df["pt_names"].unique()
    data_folder = "./data"
    params = [(data_folder, pt_name, col, df[df["pt_names"] == pt_name], value) for pt_name in pt_names]
    with Pool(32) as p:
        ttest_result = p.starmap(retrive_one_patient, params)
    pos_data = []
    neg_data = []
    for r in ttest_result:
        if r[0].shape[0] != 0: 
            pos_data.append(r[0])
        if r[1].shape[0] != 0:
            neg_data.append(r[1])
    return np.concatenate(pos_data), np.concatenate(neg_data)

def t_test_all_samples(df, col, value):
    pos_data, neg_data = retrive_patients(df, col, value)
    pos_data = pos_data.reshape(pos_data.shape[0], -1)
    neg_data = neg_data.reshape(neg_data.shape[0], -1)
    ttest = stats.ttest_ind(pos_data, neg_data, axis=0, equal_var=False, alternative="greater")[1]
    ttest = ttest.reshape(1, 64, 64)
    res = (ttest < 0.01).astype(int)
    return res
def plot(data, df, folder):
    os.makedirs(folder, exist_ok=True)
    if data.shape[0] > 0:
        plt.figure()
        plt.imshow(np.mean(data, axis=0))
        pt_name = df.iloc[0]["pt_names"]
        plt.savefig(f"{folder}/{pt_name}.png")
        plt.close()

def ttest_one_patent(data_folder, pt_name, col, df, value):
    pt_df, data = load_patient(data_folder, pt_name)
    pt_df = pt_df[["pt_names", "channel_name", "start", "end", "index"]]
    df.drop(columns=["index"], inplace=True)
    pt_df = pt_df.merge(df, on=["pt_names", "channel_name", "start", "end"])
    pos_index = pt_df[pt_df[col] == value]["index"].values
    neg_index = pt_df[pt_df[col] != value]["index"].values
    print(pt_df[col].unique(), value, "pos", pos_index.shape, "neg", neg_index.shape)
    pos_data = data[pos_index]
    return ttest(data, pos_index, neg_index), pos_data

def ttest_patients(df, col, value):
    pt_names = df["pt_names"].unique()
    data_folder = "./data"
    params = [(data_folder, pt_name, col, df[df["pt_names"] == pt_name], value) for pt_name in pt_names]
    with Pool(32) as p:
        ttest_result = p.starmap(ttest_one_patent, params)
    res, pos_data = zip(*ttest_result)
    res_res, pos_data_res = [], []
    for i in range(len(res)):
        r, p = res[i], pos_data[i]
        if len(p) != 0:
            res_res.append(r)
            pos_data_res.append(p)
    return np.array(res_res), np.concatenate(pos_data_res)

def ttest_all(df, col, savefn):
    df_ = df.dropna(subset=[col])
    anatomicals = df_[col].unique()
    # make then upper case in first letter
    anatomicals = sorted([anatomical.capitalize() for anatomical in anatomicals if anatomical != "NaN"])
    #print the df["anatomy"].unique() and counts
    fig, axs = plt.subplots(1,len(anatomicals), figsize=(3*len(anatomicals), 4.5), sharey=True)
    res, pos_data = [], []
    for i in range(len(anatomicals)):
        if t_test_type == "patient":
            t_test_result, pos = ttest_patients(df_, col, anatomicals[i].lower())
            mean_t_test = np.mean(t_test_result, axis=0)
        else:
            t_test_result = t_test_all_samples(df_, col, anatomicals[i].lower())
            mean_t_test = t_test_result[0]
        res.append(mean_t_test)
        pos_data.append(pos)
        axs[i].imshow(mean_t_test)
        axs[i].set_title(f"{anatomicals[i]} {len(t_test_result)} subjects")
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
    # super title
    pos_data = np.asarray(pos_data, dtype="object")
    np.savez_compressed(savefn.replace(".png", ".npz"), res=res, anatomicals=anatomicals, pos_data=pos_data)
    plt.suptitle(f"T-test of demographic vs non demographic in different {col}", fontsize=20)
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()


if __name__ == "__main__":
    t_test_type = sys.argv[1]
    suffix = sys.argv[3]
    name = sys.argv[2]
    paths = glob.glob(f"./res/{name}/fold_*/{suffix}")
    embedding = "train_embedding.csv"
    label_path = "train_overall_.npz"
    cols = ["anatomy"]
    for soz in ["SOZ", "non-SOZ"]:
        for col in cols:
            df_all = []
            for path in paths:
                df_test = create_df(path,label_path.replace("train","test"),embedding.replace("train","test"))
                df_all.append(df_test)
            df_all = pd.concat(df_all)
            df_all = df_all[df_all["vae-predict"]!="Artifact"]
            if soz == "SOZ":
                df_all = df_all[df_all["SOZ"] == 1]
                save_fn = os.path.join(f"./res/{name}/ttest_annatomical_soz_non-norm" ,f"ttest_{col}_pair.png")
            else:
                df_all = df_all[df_all["seizure-free"] == 1]
                df_all = df_all[df_all["removed"] != 1]
                save_fn = os.path.join(f"./res/{name}/ttest_annatomical_none_soz_non-norm" ,f"ttest_{col}_pair.png")
            df_all["pred"] = df_all["vae-predict"].replace({"Physiological":0,"Pathological":1})
            os.makedirs(os.path.dirname(save_fn), exist_ok=True)
            ttest_all(df_all, col, save_fn)
    
