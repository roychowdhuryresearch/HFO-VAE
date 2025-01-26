import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fit_embedding import sample
from load_embedding import create_df
import matplotlib.lines as mlines
def plot_male(df, savefn, n_sample_df = 200, total = 10000):
    #df,_ = sample(df, None, n_sample_df)
    fig = plt.figure(figsize=(7, 7))
    df = df[df["vae-predict"]!="Artifact"]
    df_male = df[df["Male"]==1]
    df_female = df[df["Male"]!=1]
    df_male = sample(df_male, None, int(total/2//len(df_male["pt_names"].unique())))[0]
    df_female = sample(df_female, None, int(total/2//len(df_female["pt_names"].unique())))[0]
    legend_handles = []
    scatter = plt.scatter(df_male["Projected Dimension 1"], df_male["Projected Dimension 2"], s=10, alpha=0.3,label="Male")
    legend_handles.append(mlines.Line2D([], [], color=scatter.get_facecolor()[0], marker='o', linestyle='None', markersize=10,label="Male", alpha=1))
    scatter = plt.scatter(df_female["Projected Dimension 1"], df_female["Projected Dimension 2"], s=10, alpha=0.3,label="Female")
    legend_handles.append(mlines.Line2D([], [], color=scatter.get_facecolor()[0], marker='o', linestyle='None', markersize=10,label="Female", alpha=1))
    plt.legend(handles = legend_handles, fontsize=15)
    plt.axis('off')
    plt.title("Gender", fontsize=20)
    plt.savefig(savefn)
    plt.close()

def draw(path, embedding, label_path):
    os.makedirs(path+"/embedding_figs", exist_ok=True)
    df_train = create_df(path,label_path,embedding)
    df_test = create_df(path,label_path.replace("train","test"),embedding.replace("train","test"))
    df = pd.concat([df_train,df_test])
    plot_male(df, path+"/embedding_figs/male.png")
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
        param_list.append((path, embedding, label_path))
    with mp.Pool(5) as p:
        p.starmap(draw, param_list)


