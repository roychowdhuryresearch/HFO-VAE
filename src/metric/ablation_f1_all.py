import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def get_f1(date, suffix):
    n_artifact = suffix.split("_")[0]
    n_mphfo = suffix.split("_")[1]
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
            clf = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=None, criterion='gini', class_weight='balanced') 
            #clf = RandomForestClassifier(random_state=0, n_estimators=100, class_weight='balanced', max_depth=7)
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
    # scatter plot
    mean_df = df[df["t"] == "f1"]
    sns.scatterplot(x="Sample", y=col, data=mean_df, ax=ax, markers="X")
    std_df = df[df["t"] == "f1_std"]
    ax.errorbar(x=np.arange(len(mean_df)), y=mean_df[col], yerr=std_df[col]/5, fmt="none", capsize=5, color="black")
    # baseline auc
    baseline_f1 = mean_df[base_col].mean()
    baseline_std = std_df[base_col].mean()
    ax.axhline(y=baseline_f1, color='red', linestyle='--', label=base_col)
    # draw shaded area using std 
    ax.fill_between(np.arange(len(mean_df)), baseline_f1 - baseline_std/5, baseline_f1 + baseline_std/5, alpha=0.2, color='blue')
    ax.set_xlabel("Num. Samp. in Arifact Reject./Num. Samp. in mpHFO Detect.", fontsize=fontsize)
    ax.set_ylabel("F1", fontsize=fontsize)
    #ax.set_ylim([0.55, 0.85])   
    ax.legend(loc = "lower left", fontsize=fontsize)
    # ticks
    ax.set_xticklabels(mean_df["Sample"], rotation=45, ha='right', fontsize=fontsize, rotation_mode="anchor")
    ax.tick_params(axis='both', which='major', labelsize=fontsize, pad=0)

if __name__ == "__main__":
    date = sys.argv[1]
    fn = f"res/{date}/RF_predict_all_{date}.csv"
    if not os.path.exists(fn):
        folder = f"./res/{date}"
        res_folders = glob.glob(folder + "/*_81")
        def extract_f1(folder):
            res = []
            suffix = folder.split("/")[-1]
            res.append(get_f1(date, suffix))
            return pd.concat(res)
        df = pd.concat([extract_f1(folder) for folder in res_folders])
        df.to_csv(fn)
        
    df = pd.read_csv(fn)

    df["n_artifact"] = df["n_artifact"].astype(int)/1000
    df["n_mphfo"] = df["n_mphfo"].astype(int)/1000
    df["Sample"] = df["n_artifact"].astype(str) + "k/" + df["n_mphfo"].astype(str) + "k"
    # reset index column, name it as t 
    df = df.rename(columns={"Unnamed: 0":"t"})
    sns.set_theme()
    fig, axs = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
    # sns theme
    frontsize = 8
    for ax in axs:
        # grid
        ax.grid(True)
        ax.tick_params(rotation=45,axis='x', pad=0.1, labelsize=frontsize)
    df.rename(columns={"% Path. HFO":"mpHFO", "% spk-HFO":"spkHFO", "SOZ Resc. + Age + Gender + % spk-HFO":"demo.+soz+spkHFO", "SOZ Resc. + Age + Gender + % Path. HFO":"demo.+soz+mpHFO"}, inplace=True)
    draw_f1(df, "mpHFO", axs[0], fontsize=frontsize, base_col="spkHFO")
    draw_f1(df, "demo.+soz+mpHFO", axs[1], fontsize=frontsize, base_col="demo.+soz+spkHFO")
    axs[0].set_title("F1 mpHFO", fontsize=frontsize)
    axs[1].set_title("F1 base.+soz+mpHFO", fontsize=frontsize)
    # pad x tick labels
    plt.tight_layout()
    plt.savefig(f"./fig/{date}_f1_all_ablation.png", dpi=300)
