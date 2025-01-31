import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from fit_embedding import sample
import matplotlib.lines as mlines
from scipy.stats import ttest_ind
# no user warning
import warnings
warnings.filterwarnings("ignore")
import glob
from scipy.stats import wilcoxon
def process_acc(ax, column, fontsize=6):
    acc, acc_random = [], []
    fns = glob.glob(f'/mnt/SSD5/lawrence/VAE/fig/compare_stats/fold_*/{column}_acc_2.npz') # These files are generated by fig_code/embedding_pred.py
    for fn in fns:
        data = np.load(fn)
        acc.append(data['acc_real'])
        acc_random.append(data['acc_random'])
    acc = np.array(acc).reshape(-1)
    acc_random = np.array(acc_random).reshape(-1)
    # size of the boxplot is 0.2
    box_positions = [0.0, 1]  # Adjust these values as needed
    # positions = np.arange(2) * 0.5
    data = pd.DataFrame({'value': acc.tolist() + acc_random.tolist(), 
                         'type': ['acc'] * len(acc) + ['acc_random'] * len(acc_random)})
    sns.boxplot(data=data, x='type', y='value', ax=ax, width=0.4, fliersize=0.5, linewidth=0.5, palette='Set2')
    #sns.boxplot(data=[acc,acc_random], ax=ax, width=0.4, fliersize=0.5, linewidth=0.5, palette='Set2', positions=box_positions)
    #ax.set_xticks(box_positions)
    print(column, acc.mean(),acc.std() , acc_random.mean(), acc_random.std())
    ax.set_xticklabels(['Real', 'Random'], fontsize=fontsize, rotation=45)
    ax.set_ylabel('Accuracy', fontsize=fontsize)
    ax.set_xlabel('')

    # y ticks are 0.1 apart with font size 6
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    p_val_ttest = ttest_ind(acc, acc_random, alternative='greater')[1]
    # p_val_wilcoxon = wilcoxon(acc, acc_random, alternative='greater')[1]
    #p_val_ttest = (np.sum(acc<acc_random) + 1) / (len(acc) + 1)
    #ax.set_title(f'p-value: t-test: {p_val_ttest:.2f}, wilcoxon: {p_val_wilcoxon:.2f}', fontsize=fontsize)
    bottom, top = ax.get_ylim()
    y_range = top - bottom
    bar_height = top + (y_range * 0.07)  
    bar_tips = bar_height - (y_range * 0.02)
    bar_width = 0.01
    ax.set_ylim([0, bar_height + (y_range * 0.1)])
    # plot the bar
    ax.plot([box_positions[0], box_positions[0]], [bar_height-2*bar_width, bar_tips-2*bar_width], linewidth=1, color='k')
    ax.plot([box_positions[1], box_positions[1]], [bar_height-2*bar_width, bar_tips-2*bar_width], linewidth=1, color='k')
    ax.plot([box_positions[0], box_positions[1]], [bar_tips, bar_tips], linewidth=1, color='k')
    ax.text(0.5, bar_height + 0.02, f'p={p_val_ttest:.2f}', fontsize=fontsize, horizontalalignment='center', verticalalignment='center')
    #ax.set_ylim([bottom, top + (y_range * 0.1)])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([-0.5, 1.5])
    # Hide the right and top ticks
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

def plot_dataset(df, ax, fontsize, total = 10000):
    df = df.copy()
    df["dataset"] = df["dataset"].replace("seeg", "UCLA SEEG")
    df["dataset"] = df["dataset"].replace("ucla", "UCLA grid/strip")
    df["dataset"] = df["dataset"].replace("detroit", "Detroit grid/strip")
    dataset = df["dataset"].unique()
    df = df[df["vae-predict"]!="Artifact"]
    legend_handles = []
    for i in range(len(dataset)):
        df_ = df[df["dataset"]==dataset[i]] 
        df_,_ = sample(df_, None, total//len(dataset)//len(df_["pt_names"].unique()))
        scatter = ax.scatter(df_["Projected Dimension 1"], df_["Projected Dimension 2"], s=2, alpha=0.1,label=f"{dataset[i]}")
        legend_handle = mlines.Line2D([], [], color=scatter.get_facecolor()[0], marker='o', linestyle='None', markersize=2,label=f"{dataset[i]}", alpha=1)
        legend_handles.append(legend_handle)
    ax.legend(handles = legend_handles, fontsize=fontsize, loc="lower left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis('off')
    # make it square
    ax.set_aspect('equal', 'box')

def plot_pathology(df, ax, fontsize, total = 10000):
    df = df.copy()
    df = df[df["resection-status"] ==1]
    names = ["HS", "FCD", "Tumor", "Others"]
    df["pathology"] = df["pathology"].replace("tumor", "Tumor")
    df["pathology"] = df["pathology"].apply(lambda x: x if x in names else "Others")
    df = df[df["vae-predict"]!="Artifact"]
    legend_handles = []
    for i in range(len(names)):
        df_ = df[df["pathology"]==names[i]] 
        df_,_ = sample(df_, None, total//len(names)//len(df_["pt_names"].unique()))
        scatter = ax.scatter(df_["Projected Dimension 1"], df_["Projected Dimension 2"], s=2, alpha=0.1,label=f"{names[i]}")
        legend_handle = mlines.Line2D([], [], color=scatter.get_facecolor()[0], marker='o', linestyle='None', markersize=2,label=f"{names[i]}", alpha=1)
        legend_handles.append(legend_handle)
    ax.legend(handles = legend_handles, fontsize=fontsize, loc="lower left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis('off')
    # make it square
    ax.set_aspect('equal', 'box')

def plot_gender(df, ax, fontsize, total = 10000):
    df = df.copy()
    names = ["Female", "Male"]
    df["Male"] = df["Male"].replace(1, "Male")
    df["Male"] = df["Male"].replace(0, "Female")
    df = df[df["vae-predict"]!="Artifact"]
    legend_handles = []
    for i in range(len(names)):
        df_ = df[df["Male"]==names[i]] 
        df_,_ = sample(df_, None, total//len(names)//len(df_["pt_names"].unique()))
        scatter = ax.scatter(df_["Projected Dimension 1"], df_["Projected Dimension 2"], s=2, alpha=0.1,label=f"{names[i]}")
        legend_handle = mlines.Line2D([], [], color=scatter.get_facecolor()[0], marker='o', linestyle='None', markersize=2,label=f"{names[i]}", alpha=1)
        legend_handles.append(legend_handle)
    ax.legend(handles = legend_handles, fontsize=fontsize, loc="lower left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis('off')
    # make it square
    ax.set_aspect('equal', 'box')

def plot_age(df, ax, fontsize, total = 10000):
    df = df.copy()
    age_ranges = [0,6,11,16,21,float("inf")]
    df["age"] = pd.cut(df["Age_yr"], age_ranges, labels=["0-5", "6-10", "11-15", "16-20", "21+"])
    names = ["0-5", "6-10", "11-15", "16-20", "21+"]
    df = df[df["vae-predict"]!="Artifact"]
    legend_handles = []
    for i in range(len(names)):
        df_ = df[df["age"]==names[i]] 
        df_,_ = sample(df_, None, total//len(names)//len(df_["pt_names"].unique()))
        scatter = ax.scatter(df_["Projected Dimension 1"], df_["Projected Dimension 2"], s=2, alpha=0.1,label=f"{names[i]}")
        legend_handle = mlines.Line2D([], [], color=scatter.get_facecolor()[0], marker='o', linestyle='None', markersize=2,label=f"{names[i]}", alpha=1)
        legend_handles.append(legend_handle)
    ax.legend(handles = legend_handles, fontsize=fontsize, loc="lower left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.axis('off')
    # make it square
    ax.set_aspect('equal', 'box')

if __name__ == "__main__":
    # gridspec inside gridspec
    df = pd.read_csv("/mnt/SSD5/lawrence/VAE/figure3_3.csv")
    df = df.groupby("pt_names").head(200)
    fig = plt.figure(layout='constrained', figsize=(6, 5))
    subfigs = fig.subfigures(2, 2)
    funcs = {"dataset":plot_dataset,
            "Male":plot_gender,
            "Age_yr":plot_age,
            "pathology":plot_pathology}
    labels = {"dataset":"Recording sites/types",
            "Male":"Sex", 
            "Age_yr":"Age groups",
                "pathology":"pathologies"} 
    key = list(labels.keys())
    fontsize = 6
    ax_list = []
    for i, s in enumerate(subfigs.ravel()):
        axs = s.subplots(1, 2, width_ratios=[2, 0.5])
        funcs[key[i]](df, axs[0], 6, total=5000)
        process_acc(axs[1], key[i])
        s.suptitle(labels[key[i]], fontsize=fontsize+2, weight='bold')
        ax_list.append(axs[0])
        ax_list.append(axs[1])
    # add abcd to each subfigure
    for i, ax in enumerate(ax_list):
        ax.text(-0.1, 1.1, chr(ord('a') + i), transform=ax.transAxes, 
                size=fontsize+2, weight='bold')

    plt.savefig("/mnt/SSD5/lawrence/VAE/fig/figure3.jpg", dpi=300)