import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fit_embedding import sample
from load_embedding import create_df
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines

def plot_pathology(df, savefn, n_sample_df = 4000, total = 10000):
    df = df[df["removed"]==1]
    df = df[df["vae-predict"]!="Artifact"]
    df["pathology"] = df["pathology"].replace("tumor", "Tumor")
    pathologies = ("FCD","Tumor", "HS")
    
    fig = plt.figure(figsize=(7, 7))
    legend_handles = []
    for i in range(len(pathologies)):
        df_in_pathology = df[df["pathology"]==pathologies[i]]
        df_in_pathology,_ = sample(df_in_pathology, None, total//len(pathologies)//len(df_in_pathology["pt_names"].unique()))
        scatter = plt.scatter(df_in_pathology["Projected Dimension 1"], df_in_pathology["Projected Dimension 2"], s=10, alpha=0.3,label=pathologies[i])
        legend_handle = mlines.Line2D([], [], color=scatter.get_facecolor()[0], marker='o', linestyle='None', markersize=10,label=f"{pathologies[i]}", alpha=1)
        legend_handles.append(legend_handle)
    
    plt.legend(handles = legend_handles, fontsize=15)
    plt.title("Pathology", fontsize=20)
    plt.axis('off')    
    plt.savefig(savefn)
    plt.close()

def draw(path, label_path, embedding, savefn):
    os.makedirs(path+"/embedding_figs", exist_ok=True)
    df_train = create_df(path,label_path,embedding)
    df_test = create_df(path,label_path.replace("train","test"),embedding.replace("train","test"))
    plot_pathology(pd.concat([df_train,df_test]),savefn)
    print("done")

if __name__ == "__main__":
    #paths:
    name = sys.argv[1]
    suffix = sys.argv[2]
    paths = glob.glob(f"./res/{name}/fold_*/{suffix}")
    embedding = "train_embedding.csv"
    label_path = "train_overall_.npz"

    import multiprocessing as mp
    param_list = []
    for path in paths:
        param_list.append((path, label_path, embedding, path+"/embedding_figs/pathology.png"))
    with mp.Pool(5) as p:
        p.starmap(draw, param_list)