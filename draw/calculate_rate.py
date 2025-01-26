import os
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import src.param as param
# disable warnings
import warnings
warnings.filterwarnings("ignore")

def processing(name, suffix):
    folder = f"./res/{name}/"
    fns = glob.glob(os.path.join(folder, "fold_*", f"{suffix}/test_embedding.csv"))
    agree = 0
    total = 0
    df_final = []
    for fn in fns:
        df = pd.read_csv(fn)
        df["spike_pred"] = ((df["artifact"] > 0.5) & (df["spike"] > 0.5)).astype(int)
        df["spike_pred"][df["artifact"] < 0.5] = -1
        df_final.append(df)
    df_final = pd.concat(df_final)

    df_pred = df_final[df_final["pred"] != -1]
    df_pred["pred"] = df_pred["pred"].copy().astype(int)
    df_pred["soz"] = df_pred["soz"].copy().astype(int)
    df_soz = df_pred.groupby(["pt_names", "channel_name" ,"soz"]).agg({"pred": ["sum", "count"]}).reset_index()
    df_soz.columns = ["pt_names","channel_name" ,"soz", "n_mpHFO", "n_HFO"] 

    df_meta = pd.read_csv(param.get_args()["meta_fn"])

    df_soz = df_soz.merge(df_meta, left_on="pt_names", right_on="pt_name", how="left")
    df_soz["r_non_mpHFO"] = (df_soz["n_HFO"] - df_soz["n_mpHFO"])/df_soz["length"]*60
    df_soz["r_mpHFO"] = df_soz["n_mpHFO"]/df_soz["length"]*60

    df_temp = df_soz[["pt_names", "soz", "r_non_mpHFO","dataset"]].copy()
    df_temp["name"] = "non_mpHFO"
    df_temp.columns = ["pt_names", "soz", "rate","dataset", "name"]
    df_temp2 = df_soz[["pt_names", "soz", "r_mpHFO", "dataset"]].copy()
    df_temp2["name"] = "mpHFO"
    df_temp2.columns = ["pt_names", "soz", "rate", "dataset", "name"]
    df_soz = pd.concat([df_temp, df_temp2])
    # info is in dataset + soz 
    df_soz["info"] = df_soz["dataset"] + df_soz["soz"].astype(str)

    df_soz = df_soz.groupby(["info","pt_names","soz","name"]).agg({"rate": "mean"}).reset_index()
    df_soz.to_csv(f"./res/{name}/df_soz.csv", index=False)

    sns.boxplot(data=df_soz, x="rate", y="info", hue="name", orient="h")
    plt.savefig(f"./fig/{name}_df_soz.png", dpi=300)

if __name__ == "__main__":
    name = sys.argv[1]
    suffix = sys.argv[2]
    processing(name, suffix)
