import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines
# color map 
import matplotlib.cm as cm


def plot_auc_roc(ax, fontsize=6):
    biomarkers = ["'Age_yr', 'Male', 'soz_resected'", "'soz_resected', 'Age_yr', 'Male', 'r_spike'", "'soz_resected', 'Age_yr', 'Male', 'r_pred'"]
    text = ["base.+soz", "base.+soz\n+spkHFO", "base.+soz\n+mpHFO"]
    for i, b in enumerate(biomarkers):
        loaded = np.load(f"./res/{date}/{suffix}/roc/[{b}]_roc.npz")
        fpr = loaded["fpr"]
        tpr = loaded["tpr"]
        auc_value = loaded["auc"]
        ax.plot(fpr, tpr, linewidth=2, alpha=0.8, label=f"{text[i]} ({auc_value:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=2, alpha=0.8)
    ax.legend(loc="lower right", fontsize=fontsize)
    ax.set_xlabel("False Positive Rate", fontsize=fontsize, labelpad=0.1)
    #ax.set_ylabel("True Positive Rate", fontsize=fontsize)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=fontsize, pad=0.1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.5)

def plot_auc_roc1(ax, fontsize=6):
    biomarkers = ["hfo", "spike", "pred"]
    text = ["HFO", "spkHFO", "mpHFO"]
    for i, b in enumerate(biomarkers):
        loaded = np.load(f"./res/{date}/{suffix}/roc/['r_{b}']_roc.npz")
        fpr = loaded["fpr"]
        tpr = loaded["tpr"]
        auc_value = loaded["auc"]
        ax.plot(fpr, tpr, linewidth=2, alpha=0.8, label=f"{text[i]} ({auc_value:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=2, alpha=0.8)
    ax.legend(loc="lower right", fontsize=fontsize)
    ax.set_xlabel("False Positive Rate", fontsize=fontsize, labelpad=0.1)
    ax.set_ylabel("True Positive Rate", fontsize=fontsize, labelpad=0.1)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.tick_params(axis='both', which='major', labelsize=fontsize, pad=0.1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.5)

def draw_f1(ax, fontsize=6):
    df = pd.read_csv(f"res/{date}/{suffix}/RF_predict_{date}_{suffix}.csv")
    # only remove the second row
    df.drop(df.index[1], inplace=True)
    df = df.iloc[[0,1,2,4,5,6,7,8]]
    df = df.iloc[[0,1,2,3,6,7]].copy()
    category_order = df['feature'].unique()
    sns.barplot(x="feature", y="f1_mean", data=df, ax=ax, order=category_order, width=0.4, hue="feature", hue_order=category_order, palette="Set2")
    ax.errorbar(x=np.arange(len(df)), y=df["f1_mean"], yerr=df["f1_std"]/5, fmt="none", capsize=5, color="black") 
    # remove the third row 
    labels = ["HFO", "spkHFO", "mpHFO", "demo.","+soz","+spkHFO","+mpHFO", "+spkHFO\n+soz","+mpHFO\n+soz"]
    labels = ["HFO", "spkHFO", "mpHFO", "base.\n+soz","base.\n+spkHFO","base.\n+mpHFO", "base.\n+spkHFO\n+soz","base.\n+mpHFO\n+soz"]
   
    index = np.array([0,1,2,3,6,7])
    labels = np.array(labels)[index]
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
    ax.set_ylabel("F1 score", fontsize=fontsize)
    ax.set_xlabel("")
    ax.set_ylim([0.6, 0.88])
    ax.tick_params(axis='both', which='major', labelsize=fontsize, pad=0.1)
    # draw vertical line
    #ax.axvline(x=2.5, color="black", linewidth=1, linestyle="--")
    #ax.axvline(x=6.5, color="black", linewidth=1, linestyle="--")
    ax.grid(True, alpha=0.5)
    print()


if __name__ == "__main__":
    date = "2023-12-28_2"
    suffix = "10000_2000_81"
    #df = pd.read_csv(f'./res/{date}/{suffix}/embedding_save.csv')
    cm = 1/2.54 
    #fig = plt.figure(figsize=(10.1*cm, 15*cm))
    fontsize = 6
    fig = plt.figure(layout='constrained', figsize=(9*cm, 9*cm)) # width = 23cm, height = 20cm
    # two rows, 1 column
    subfigs = fig.subfigures(2, 1, width_ratios=[1], height_ratios=[1, 1])
    ax8, ax9 = subfigs[0].subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
    #ax1 = subfigs[1].subplots(1, 1)
    ax10 = subfigs[1].subplots(1, 1)

    plot_auc_roc1(ax8, fontsize=fontsize)
    plot_auc_roc(ax9, fontsize=fontsize)
    draw_f1(ax10, fontsize=fontsize)
    #plot_header(ax1)
    #plt.tight_layout()
    # add abcdef for each axs
    fig.text(0.02, 0.97, "a", ha='center', fontsize=fontsize+2, weight='bold')
    fig.text(0.53, 0.97, "b", ha='center', fontsize=fontsize+2, weight='bold')
    fig.text(0.02, 0.5, "c", ha='center', fontsize=fontsize+2, weight='bold')
    #fig.text(0.02, 0.33, "d", ha='center', fontsize=fontsize+2, weight='bold')
    # fig.subplots_adjust(hspace=0.1) 
    plt.savefig("./fig/figure6_aes.jpg", dpi=300)