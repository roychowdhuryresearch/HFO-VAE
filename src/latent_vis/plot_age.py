import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import pandas as pd
import matplotlib.lines as mlines
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fit_embedding import sample
from load_embedding import create_df

def plot_age(df, savefn, n_sample_df = 5000, total = 10000):
    #df drop the artifact
    df = df[df["vae-predict"]!="Artifact"]
    legend_handles = []
    age_ranges = [0,6,11,16,21,float("inf")]
    fig = plt.figure(figsize=(7, 7))
    for i in range(len(age_ranges)-1):
        df_in_age = df[(df["Age_yr"]>=age_ranges[i]) & (df["Age_yr"]<age_ranges[i+1])]
        df_in_age,_ = sample(df_in_age, None, total//len(age_ranges)//len(df_in_age["pt_names"].unique()))
        scatter = plt.scatter(df_in_age["Projected Dimension 1"], df_in_age["Projected Dimension 2"], s=10, alpha=0.3,label=f"Age {age_ranges[i]}-{age_ranges[i+1]}" if i!=len(age_ranges)-2 else f"Age {age_ranges[i]}-")
        legend_handle = mlines.Line2D([], [], color=scatter.get_facecolor()[0], marker='o', linestyle='None', markersize=10,label=f"Age {age_ranges[i]}-{age_ranges[i+1]}" if i!=len(age_ranges)-2 else f"Age {age_ranges[i]}-", alpha=1)
        legend_handles.append(legend_handle)
    plt.legend(handles = legend_handles, fontsize=15)
    plt.title("Age", fontsize=20)
    plt.axis('off')
    plt.savefig(savefn)
    plt.close()

def draw(path, label_path, embedding, savefn):
    os.makedirs(path+"/embedding_figs", exist_ok=True)
    df_train = create_df(path,label_path,embedding)
    df_test = create_df(path,label_path.replace("train","test"),embedding.replace("train","test"))
    plot_age(pd.concat([df_train,df_test]),savefn)
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
    param_list = []
    for path in paths:
        param_list.append((path, label_path, embedding, path+"/embedding_figs/age.png"))
    with mp.Pool(5) as p:
        p.starmap(draw, param_list)