import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fit_embedding import sample
from load_embedding import create_df
def plot_dataset(df, savefn, n_sample_df = 500, total = 10000):

    df.dropna(subset=["dataset"], inplace=True)
    # replace ucla as UCLA Grid&Strip
    df["dataset"] = df["dataset"].replace("ucla", "UCLA Grid&Strip")
    df["dataset"] = df["dataset"].replace("seeg", "SEEG")
    df["dataset"] = df["dataset"].replace("detroit", "Detroit Grid&Strip")
    datasets = ("UCLA Grid&Strip","SEEG", "Detroit Grid&Strip")
    df = df[df["vae-predict"] != "Artifact"]
    legend_handles = []
    fig = plt.figure(figsize=(7, 7))
    for i in range(len(datasets)):
        df_in_pathology = df[df["dataset"]==datasets[i]]
        df_in_pathology,_ = sample(df_in_pathology, None, total//len(datasets)//len(df_in_pathology["pt_names"].unique()))
        # shuffle and pick 4000 with seed 0
        scatter = plt.scatter(df_in_pathology["Projected Dimension 1"], df_in_pathology["Projected Dimension 2"], s=10, alpha=0.3,label=datasets[i])
        legend_handle = mlines.Line2D([], [], color=scatter.get_facecolor()[0], marker='o', linestyle='None', markersize=10,label=f"{datasets[i]}", alpha=1)
        legend_handles.append(legend_handle)
    plt.legend(handles = legend_handles, fontsize=15)
    plt.title("Dataset", fontsize=20)
    plt.axis('off')
    plt.savefig(savefn)
    plt.close()

def draw(path, label_path, embedding, savefn):
    os.makedirs(path+"/embedding_figs", exist_ok=True)
    df_train = create_df(path,label_path,embedding)
    df_test = create_df(path,label_path.replace("train","test"),embedding.replace("train","test"))
    fold_num = path.split("/")[-2].split("_")[-1]   
    pd.concat([df_train,df_test]).to_csv(f"figure3_{fold_num}.csv")
    plot_dataset(pd.concat([df_train,df_test]),savefn)
    print("done")

if __name__ == "__main__":
    #paths:
    import sys
    name = sys.argv[1]
    suffix = sys.argv[2]
    paths = glob.glob(f"./res/{name}/fold_*/{suffix}")
    embedding = "train_embedding.csv"
    label_path = "train_overall_.npz"
    import multiprocessing as mp
    from functools import partial
    param_list = []
    for path in paths:
        param_list.append((path, label_path, embedding, path+"/embedding_figs/dataset.png"))
    with mp.Pool(5) as p:
        p.starmap(draw, param_list)