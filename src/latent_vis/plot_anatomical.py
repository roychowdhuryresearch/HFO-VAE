import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import os

import pandas as pd
import matplotlib.lines as mlines
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fit_embedding import sample
from load_embedding import create_df

def plot_anatomical(df, savefn, n_sample_df = 4000, total = 10000):
    #df,_ = sample(df, None, 10000)
    # replace frontal as Frontal
    df["anatomy"] = df["anatomy"].replace("frontal", "Frontal")
    df["anatomy"] = df["anatomy"].replace("temporal", "Temporal")
    df["anatomy"] = df["anatomy"].replace("occipital", "Occipital")
    df["anatomy"] = df["anatomy"].replace("parietal", "Parietal")
    df["anatomy"] = df["anatomy"].replace("limbic", "Limbic")
    anatomicals = ("Frontal","Temporal", "Occipital", "Parietal", "Limbic")
    df = df[df["vae-predict"]!="Artifact"]
    fig = plt.figure(figsize=(7, 7))
    legend_handles = []
    for i in range(len(anatomicals)):
        df_ = df[df["anatomy"]==anatomicals[i]]
        df_,_ = sample(df_, None, total//len(anatomicals)//len(df_["pt_names"].unique()))
        scatter = plt.scatter(df_["Projected Dimension 1"], df_["Projected Dimension 2"], s=10, alpha=0.3,label=f"{anatomicals[i]}")
        legend_handle = mlines.Line2D([], [], color=scatter.get_facecolor()[0], marker='o', linestyle='None', markersize=10,label=f"{anatomicals[i]}", alpha=1)
        legend_handles.append(legend_handle)
    plt.legend(handles = legend_handles, fontsize=15)
    plt.title("Anatomical", fontsize=20)
    plt.axis('off')
    plt.savefig(savefn)
    plt.close()
 

def draw(path, label_path, embedding, savefn):
    os.makedirs(path+"/embedding_figs", exist_ok=True)
    df_train = create_df(path,label_path,embedding)
    df_test = create_df(path,label_path.replace("train","test"),embedding.replace("train","test"))
    plot_anatomical(pd.concat([df_train,df_test]),savefn)
    print("done")

if __name__ == "__main__":
    import sys
    name = sys.argv[1]
    suffix = sys.argv[2]
    paths = glob.glob(f"./res/{name}/fold_*/{suffix}")
    embedding = "train_embedding.csv"
    label_path = "train_overall_.npz"

    import multiprocessing as mp
    param_list = []
    for path in paths:
        param_list.append((path, label_path, embedding, path+"/embedding_figs/anatomical.png"))
    with mp.Pool(5) as p:
        p.starmap(draw, param_list)
