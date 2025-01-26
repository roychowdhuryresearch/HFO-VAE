import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

def get_f1(date, suffix):
    n_artifact = int(suffix.split("_")[0])
    n_mphfo = int(suffix.split("_")[1])
    fn = f'./res/{date}/{suffix}/auc_kfold.xlsx'
    # return a dict of dataframes
    df = pd.read_excel(fn, engine='openpyxl', sheet_name=None)
    col_list = [["r_hfo"], ["r_real"],["r_spike"],["r_pred"],
                ["Age_yr", "Male","r_hfo"], ["Age_yr", "Male", "r_real"],["Age_yr", "Male", "r_spike", "n_spike"],["Age_yr", "Male", "r_pred", "n_pred"],
                ["soz_resected"], 
                ["Age_yr", "Male"],
                ["Age_yr", "Male", "r_pred"],
                ["Age_yr", "Male", "r_spike"],
                ["Age_yr", "Male", "soz_resected"],
                ["soz_resected","r_hfo"], 
                ["soz_resected","r_real"], 
                ["soz_resected","r_spike",], 
                ["soz_resected","r_pred",],
                ["soz_resected","Age_yr", "Male","r_hfo"], 
                ["soz_resected","Age_yr", "Male","r_real"], 
                ["soz_resected","Age_yr", "Male","r_spike",], 
                ["soz_resected","Age_yr", "Male","r_pred",],
                ]
    df_res = []
    for col in col_list:
        acc_list, f1_list = [], []
        for i in range(5):
            df_train = df[f"train_{i}"]
            df_test = df[f"test_{i}"]
            X_train = df_train[col].values
            y_train = df_train["seizure-free"].values
            X_test = df_test[col].values
            y_test = df_test["seizure-free"].values
            #clf = RandomForestClassifier(random_state=0, n_estimators=50, max_depth=None, criterion='gini', class_weight='balanced') 
            clf = RandomForestClassifier(random_state=42+i, n_estimators=150, class_weight='balanced', max_depth=7)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:,1]
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            acc_list.append(acc)
            f1_list.append(f1)
        df_res.append(["_".join(col),np.mean(acc_list), np.std(acc_list), np.mean(f1_list), np.std(f1_list)])
    df_final= pd.DataFrame(df_res, columns=["feature","acc_mean", "acc_std", "f1_mean", "f1_std"])
    df_res = df_final.copy()
    df_res["Acc"] = df_res.apply(lambda x: f"{x['acc_mean']:.3f}({x['acc_std']:.3f})", axis=1)
    df_res["F1"] = df_res.apply(lambda x: f"{x['f1_mean']:.3f}({x['f1_std']:.3f})", axis=1)

    df_res["feature"] = df_res["feature"].str.replace("r_hfo", "% HFO")
    df_res["feature"] = df_res["feature"].str.replace("r_real", "% Real")
    df_res["feature"] = df_res["feature"].str.replace("r_spike", "% spk-HFO")
    df_res["feature"] = df_res["feature"].str.replace("r_pred", "% Path. HFO")
    df_res["feature"] = df_res["feature"].str.replace("soz_resected", "SOZ Resc.")
    df_res["feature"] = df_res["feature"].str.replace("Age_yr", "Age")
    df_res["feature"] = df_res["feature"].str.replace("Male", "Gender")
    df_res["feature"] = df_res["feature"].str.replace("_", " + ")
    df_ratio = df_res.iloc[[0,1,2,3,9,12,11,10,-2,-1]]
    df_ratio = df_ratio[["feature", "f1_mean", "f1_std"]]
    df_ratio.rename(columns={"f1_mean":"f1"}, inplace=True)
    df_ratio.set_index("feature", inplace=True)
    df_ratio = df_ratio.transpose()
    df_ratio["n_artifact"] = n_artifact
    df_ratio["n_mphfo"] = n_mphfo
    return df_ratio

def draw_f1(df, col, ax, fontsize=6, base_col = "r_spike"):
    df = df.copy()
    df = df[df["n_artifact"] <= 10]
    df = df[df["n_mphfo"] <= 3]
    df = df.sort_values(["n_artifact", "n_mphfo"])
    # name index column as t
    df = df.reset_index()
    df = df.rename(columns={"Unnamed: 0":"t"})
    # scatter plot
    mean_df = df[df["t"] == "f1"]
    sns.scatterplot(x="Sample", y=col, data=mean_df, ax=ax, markers="X")
    std_df = df[df["t"] == "f1_std"]

    ax.errorbar(x=np.arange(len(mean_df)), y=mean_df[col], yerr=std_df[col]/5, fmt="none", capsize=5, color="black")
    # baseline auc
    baseline_f1 = mean_df[base_col].mean()
    baseline_std = std_df[base_col].mean()
    ax.axhline(y=baseline_f1, color='red', linestyle='--', label=f"F1 {base_col}")
    # draw shaded area using std 
    ax.fill_between(np.arange(-0.5,len(mean_df)+0.5), baseline_f1 - baseline_std/5, baseline_f1 + baseline_std/5, alpha=0.2, color='blue')
    ax.set_xlabel("", fontsize=fontsize)
    ax.set_ylabel("F1", fontsize=fontsize)
    #ax.set_xlim([-0.3,3.3])   
    ax.legend(loc = "lower left", fontsize=fontsize)
    # ticks
    print(mean_df["Sample"].unique())   
    ax.set_xticklabels(mean_df["Sample"], rotation=45, ha='right', fontsize=fontsize, rotation_mode="anchor")
    ax.tick_params(axis='both', which='major', labelsize=fontsize, pad=0)

def extract_auc(folder):
    fns = glob.glob(folder + "/roc/*")
    res_dict = {}
    res_dict["Num Samples in Arifact Rejection"] = int(folder.split("/")[-1].split("_")[0])
    res_dict["Num Samples in mpHFO Detection"] = int(folder.split("/")[-1].split("_")[1])
    for f in fns:
        key = f.split("/")[-1].split(".")[0].split("]")[0].replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
        auc = np.load(f)["auc"]
        res_dict[key] = auc
    return pd.DataFrame(res_dict, index=[0])

def draw_new(df, ax, baseline_auc_col = "r_spike", wanted_col="r_pred", name = "AUC mpHFO"):
    df = df.copy()

    df[name] = df[wanted_col]
    df["Num Samples in Arifact Rejection"] = df["Num Samples in Arifact Rejection"] / 1000
    df["Num Samples in mpHFO Detection"] = df["Num Samples in mpHFO Detection"] / 1000
    df = df[df["Num Samples in Arifact Rejection"] <= 10]
    df = df[df["Num Samples in mpHFO Detection"] <= 3]
    # sort by num samples in artifact rejection and mpHFO detection
    df = df.sort_values(["Num Samples in Arifact Rejection", "Num Samples in mpHFO Detection"])
    # make a col with name sample 
    # for example artifact 1000, mpHFO 1000 -> 1k/1k
    df["Sample"] = df["Num Samples in Arifact Rejection"].astype(str) + "k/" + df["Num Samples in mpHFO Detection"].astype(str) + "k"
    # draw the scatter plot
    sns.scatterplot(x="Sample", y=name, data=df, ax=ax, markers="X")
    baseline_auc = df[baseline_auc_col].mean()
    # draw the baseline auc
    if baseline_auc_col == "soz_resected,Age_yr,Male,r_spike":
        baseline_text = "AUC base.+soz+spkHFO"
    else:
        baseline_text = "AUC spkHFO"
    ax.axhline(y=baseline_auc, color='red', linestyle='--', label=baseline_text)
    ax.set_xlabel("", fontsize=frontsize)
    ax.set_ylabel("AUC", fontsize=frontsize)
    # set xticks
    print(df["Sample"].unique())
    ax.set_xticks(range(len(df["Sample"].unique())))
    ax.set_xticklabels(df["Sample"].unique(), rotation=45, ha='right', fontsize=frontsize, rotation_mode="anchor")

    # set yticks
    yticks = np.linspace(0.6, 0.75, 4)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.2f}" for y in yticks], fontsize=frontsize)
    #ax.tick_params(axis='both', which='major', labelsize=frontsize, pad=0)
    ax.legend(loc='upper right', fontsize=frontsize)
    return ax

if __name__ == "__main__":
    date = sys.argv[1]
    fn = f"res/{date}/RF_predict_all_{date}.csv"
    folder = f"./res/{date}"
    res_folders = glob.glob(folder + "/*_81")
    dfs = pd.concat([extract_auc(f) for f in res_folders])
    sns.set_theme()
    fig, axs = plt.subplots(2, 2, figsize=(6, 4), sharex=True)
    # sns theme
    frontsize = 8
    draw_new(dfs, axs[0, 0])
    draw_new(dfs, axs[0, 1], baseline_auc_col="soz_resected,Age_yr,Male,r_spike", wanted_col="soz_resected,Age_yr,Male,r_pred", name="AUC base.+soz+mpHFO")
    axs[0, 0].set_title("mpHFO", fontsize=frontsize)
    axs[0,1].set_title("base.+soz+mpHFO", fontsize=frontsize)
    axs[0,1].set_ylabel("")

    if not os.path.exists(fn):
        def extract_f1(folder):
            res = []
            suffix = folder.split("/")[-1]
            res.append(get_f1(date, suffix))
            return pd.concat(res)
        # only keep res_folders has 5000 or 10000 samples in artifact rejection and 2000 or 1000 samples in mpHFO detection
        res_folders = [f for f in res_folders if int(f.split("/")[-1].split("_")[0]) in [5000, 10000] and int(f.split("/")[-1].split("_")[1]) in [2000, 2500,3000,1500,1000]]
        print(res_folders)
        df = pd.concat([extract_f1(folder) for folder in res_folders])
        df.to_csv(fn)
    
    df = pd.read_csv(fn)

    df["n_artifact"] = df["n_artifact"].astype(int)/1000
    df["n_mphfo"] = df["n_mphfo"].astype(int)/1000
    df["Sample"] = df["n_artifact"].astype(str) + "k/" + df["n_mphfo"].astype(str) + "k"
    print(df["Sample"].unique())    
    # reset index column, name it as t 
    df = df.rename(columns={"Unnamed: 0":"t"})
    df.rename(columns={"% Path. HFO":"mpHFO", "% spk-HFO":"spkHFO", "SOZ Resc. + Age + Gender + % spk-HFO":"base.+soz+spkHFO", "SOZ Resc. + Age + Gender + % Path. HFO":"base.+soz+mpHFO"}, inplace=True)
    draw_f1(df, "mpHFO", axs[1,0], fontsize=frontsize, base_col="spkHFO")
    draw_f1(df, "base.+soz+mpHFO", axs[1,1], fontsize=frontsize, base_col="base.+soz+spkHFO")
    axs[1,0].set_title("mpHFO", fontsize=frontsize)
    axs[1,1].set_title("base.+soz+mpHFO", fontsize=frontsize)
    axs[1,1].set_ylabel("")

    frontsize = 8
    for ax in axs.flatten():
        # grid
        ax.grid(True)
        ax.tick_params(rotation=45,axis='x', pad=0.1, labelsize=frontsize)
    # add text on the bottom
    fig.text(0.5, 0.01, "Num. Samples in Training Arifact Rejection Model/mpHFO Classification Model", ha='center', fontsize=frontsize)
    # give a b c d label
    fig.text(0.01, 0.95, "a", fontsize=frontsize, weight='bold')
    fig.text(0.51, 0.95, "b", fontsize=frontsize, weight='bold')
    fig.text(0.01, 0.45, "c", fontsize=frontsize, weight='bold')
    fig.text(0.51, 0.45, "d", fontsize=frontsize, weight='bold')
    plt.tight_layout()
    plt.savefig(f"./fig/{date}_ablation_metrics.jpg", dpi=300)