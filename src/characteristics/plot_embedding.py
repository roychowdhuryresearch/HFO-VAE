import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import pandas as pd
import sys
import matplotlib.lines as mlines
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_embedding import create_df

def plot_embedding(df, n_sample_df = 5000, axs=None, fold=None):
    df = df.groupby("pt_names").head(300)
    df["vae-predict"] = df["vae-predict"].replace({"Physiological":"non-mpHFO","Pathological":"mpHFO","Artifact":"mArtifact"})
    df["spk-HFO"] = df["spk-HFO"].replace({"HFO-without-spike":"spkHFO","HFO-with-spike":"non-spkHFO","Artifact":"Artifact"})
    scatter = sns.scatterplot(data=df, x="Projected Dimension 1", y="Projected Dimension 2", hue="vae-predict", hue_order = ["non-mpHFO","mpHFO","mArtifact"], ax=axs[0], legend=True, s=2,alpha=0.1)
    scatter1 = sns.scatterplot(data=df, x="Projected Dimension 1", y="Projected Dimension 2", hue="spk-HFO", hue_order =["spkHFO","non-spkHFO","Artifact"],  ax=axs[1], legend=True, s=2,alpha=0.1)
    legend_handles = []
    labels = ["non-mpHFO","mpHFO","mArtifact"]
    # tab:10 color
    tab10 = sns.color_palette("tab10")
    for i in range(3):
        legend_handle = mlines.Line2D([], [], color=tab10[int(i)], marker='o', linestyle='None', markersize=2,label=f"{labels[i]}", alpha=1)
        legend_handles.append(legend_handle)
    scatter.legend(handles=legend_handles, loc='lower left', fontsize=6)
    legend_handles = []
    labels = ["non-spkHFO","spkHFO","Artifact"]
    for i in range(3):
        legend_handle = mlines.Line2D([], [], color=tab10[int(i)], marker='o', linestyle='None', markersize=2,label=f"{labels[i]}", alpha=1)
        legend_handles.append(legend_handle)
    scatter1.legend(handles=legend_handles, loc='lower left', fontsize=6)

    print(df["vae-predict"].unique())
    axs[0].set_title(f"Fold {fold}", fontsize=8)
    axs[0].set_aspect('equal', 'box')
    axs[1].set_aspect('equal', 'box')
    df["fold"] = fold
    # axis font size
    for ax_ in axs:
        ax_.tick_params(axis='both', which='major', labelsize=6)
    # x, y label 
    axs[0].set_xlabel("Projected Dimension 1", fontsize=6)
    axs[0].set_ylabel("Projected Dimension 2", fontsize=6)
    axs[1].set_xlabel("Projected Dimension 1", fontsize=6)
    axs[1].set_ylabel("Projected Dimension 2", fontsize=6)
    return df

    
if __name__ == "__main__":
    name = sys.argv[1]
    save_suffix = sys.argv[2]
    paths = sorted(glob.glob(f"./res/{name}/fold_*/{save_suffix}"))
    embedding = "train_embedding.csv"
    label_path = "train_overall_.npz"
    # create 2 row 5 column figure
    fig, axs = plt.subplots(2,5, figsize=(8*1.5, 6), sharey=True, sharex=True)
    df_list = []
    for i in range(len(paths)):
        df_train = create_df(paths[i],label_path,embedding)
        df_test = create_df(paths[i],label_path.replace("train","test"),embedding.replace("train","test"))
        df = pd.concat([df_train,df_test])
        df.to_csv(f"./res/{name}/{save_suffix}/embedding_{i}.csv")
        df_list.append(plot_embedding(df, axs=axs[:,i], fold=i))
    df = pd.concat(df_list)
    # save df
    df.to_csv(f'./res/{name}/{save_suffix}/embedding_save.csv')
    plt.tight_layout()
    plt.savefig(f"./fig/ablation_embedding_{name}_{save_suffix}.png", dpi=300)
