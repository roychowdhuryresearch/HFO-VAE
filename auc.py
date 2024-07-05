from sklearn.linear_model import LogisticRegression 
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
import glob
import statsmodels.api as sm
import sys
from sklearn import metrics
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor as Pool

def single_patient_agg(df_p, reconstruct_threshold):
    fold_num = df_p["fold"].unique()[0]
    pt_name = df_p["pt_names"].unique()[0]
    df_p["Resection"] = df_p["removed"]
    df_p = df_p.sort_values(["channel_name", "start"])
    df_p = process_merge(df_p, col_name="pred", option="max")
    df_p["artifact"] = df_p["artifact"] > 0.5
    df_p["spike"] = df_p["spike"] > 0.5
    pred = df_p[(df_p["pred"] ==1) & (df_p["reconstruct"] < reconstruct_threshold)].copy()
    spike = df_p[(df_p["spike"] == 1) & (df_p["artifact"] == 1)].copy()
    artifact = df_p[df_p["artifact"] == 1]
    threshold = 0
    if len(pred) <= threshold:
        r_pred = 0
        n_pred = 0
    else:
        r_pred = len(pred[pred["removed"] == 1])/ len(pred)
        n_pred = len(pred)
    if len(spike) <= threshold:
        r_spike = 0
        n_spike = 0
    else:
        r_spike = len(spike[spike["removed"] == 1])/ len(spike)
        n_spike = len(spike)
    if len(artifact) <= threshold:
        r_artifact = 0
        n_artifact = 0
    else:
        r_artifact = len(artifact[artifact["removed"] == 1])/len(artifact)
        n_artifact = len(artifact)
    if len(pred) == 0:
        n_pred_agree_spike = 0
    else:
        n_pred_agree_spike = len(pred[pred["spike"] == 1]) / len(pred)
    if len(spike) == 0:
        n_spike_agree_pred = 0
    else:
        n_spike_agree_pred = len(spike[spike["pred"] == 1]) / len(spike)
    # soz resected defined as if all soz channels are removed, the value is 1 otherwise 0
    soz_resected = 1 if len(artifact[(artifact["soz"] == 1) & (artifact["removed"] == 1)]) == len(artifact[artifact["soz"] == 1]) else 0
    # comput n_HFO per channel
    df_p = df_p[df_p["reconstruct"] < reconstruct_threshold]
    max_hfo_channel = df_p.groupby(["channel_name"]).agg({"start": "count"}).reset_index().rename(columns={"start": "n_HFO"})["n_HFO"].mean()
    
    df_pp = pd.DataFrame({"pt_name": [pt_name], "r_hfo":[len(df_p[df_p["removed"] == 1])/len(df_p)], "n_hfo": [len(df_p)],"r_real": [r_artifact], 
                        "r_spike": [r_spike], "r_pred": [r_pred], 
                        "n_real": [n_artifact], "n_spike": [n_spike], "n_pred": [n_pred],
                        "n_pred_agree_spike": [n_pred_agree_spike], "n_spike_agree_pred": [n_spike_agree_pred], "soz_resected":[soz_resected], 
                        "fold": [fold_num], "max_hfo_channel": [max_hfo_channel]
                        })
    return df_pp

def patient_agg(df, reconstruct_threshold):
    res = []

    # change to multi processing
    param_list = []
    for pt_name in df["pt_names"].unique():
        df_p = df[df["pt_names"] == pt_name].copy()
        param_list.append((df_p, reconstruct_threshold))

    with Pool(max_workers=10) as executor:
        res = executor.map(single_patient_agg, *zip(*param_list))
    return list(res)

def process_merge(df, col_name ="outputs" ,option = "mean"):
    def merge_intervals(df):
        df.sort_values("start", inplace=True)
        df["group"]=(df["start"]>df["end"].shift().cummax()).cumsum()
        df = df.groupby("group").agg({"start":"min", "end": "max", 
                                      "Resection":"max", 
                                      "soz":"max",
                                      "removed": "max",
                                      "artifact": option,
                                      "spike": option,
                                      "reconstruct": "max",
                                      col_name: option})
        return df
    df = df.sort_values(['channel_name', 'start'])
    merged_df = df.groupby(["pt_names",'channel_name'], group_keys=True).apply(merge_intervals)
    merged_df = merged_df.reset_index(allow_duplicates=True)
    return merged_df

def process_fn(fn, threshold = 0.5):
    loaded = np.load(fn, allow_pickle=True)
    df = pd.DataFrame(loaded["labels"], columns=["remove_label", "soz", "removed", "artifact", "spike"])
    df["pt_names"] = loaded["pt_names"]
    df["channel_name"] = loaded["channel_names"]
    df["start"] = loaded["starts"]
    df["end"] = loaded["ends"]
    df["detector"] = loaded["detector"]
    df["pred"] = loaded["pred"]

    df["duration"] = df["end"] - df["start"]
    df = df[df["duration"] < 150]
    df["pred_prob"] = loaded["pred_prob"]
    # all the keys in loaded
    df["reconstruct"] = loaded["reconstruct"]
    df["fold"] = fn.split("/")[-3].split("_")[-1]
    return df

def aggregate(folder, save_suffix ,threshold = 0.5,sampling="GMM"):
    if sampling == "GMM":
        fns = glob.glob(f"{folder}/fold_*/{save_suffix}/test_overall_.npz")
    else:
        fns = glob.glob(f"{folder}/fold_*/{save_suffix}/test_kmeans_new.npz")
    r = []
    for fn in fns:
        r.append(process_fn(fn, threshold))
    return pd.concat(r)
def get_pval(X, y):
    mod = sm.Logit(y,np.squeeze(X))
    fii = mod.fit(disp=False)
    p_values = fii.summary2().tables[1]['P>|z|'].to_dict()
    p_values = {k: round(v, 3) for k, v in p_values.items()}
    return p_values
# import random forest
from sklearn.ensemble import RandomForestClassifier   
from sklearn.svm import SVC
def auc_score(df, cols, save_path, plot=True):
    clf = LogisticRegression(random_state=0, max_iter=1000, penalty="l2")
    #clf = RandomForestClassifier(n_estimators=100, random_state=0, class_weight="balanced")
    clf.fit(df[cols], df["seizure-free"])
    auc = roc_auc_score(df["seizure-free"], clf.predict_proba(df[cols])[:,1])
    fpr, tpr, _ = metrics.roc_curve(df["seizure-free"],  clf.predict_proba(df[cols])[:,1])
    acc = metrics.accuracy_score(df["seizure-free"], clf.predict(df[cols]))
    # save as npz
    os.makedirs(os.path.join(save_path, "roc"), exist_ok=True)
    np.savez(f"{save_path}/roc/{cols}_roc.npz", fpr=fpr, tpr=tpr, auc=auc, acc=acc)
    if plot:
        auc_str = "{0:.3f}".format(auc)
        plt.figure(figsize=(4.2,4.2))
        plt.plot([0, 1], [0, 1],'o--')
        plt.plot(fpr,tpr,label=f"AUC="+auc_str)
        #plt.plot(fpr,tpr,label=f"AUC="+str(auc) + "\npval=" + str(p_val))
        plt.title(f"ROC curve {cols} \n acc={acc}")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc=4)
        plt.grid(True)
        plt.savefig(f"{save_path}/{cols}_roc.png")
        plt.close()
    return auc, get_pval(df[cols], df["seizure-free"]), acc

def get_auc_acc(df,folder, plot):
    #os.makedirs(folder, exist_ok=True)
    col_list = [["r_real"],["r_spike"],["r_pred"], ["soz_resected","Age_yr", "Male"], 
                ["soz_resected","Age_yr", "Male", "r_real", "n_real"], 
                ["soz_resected","Age_yr", "Male", "r_real", "n_real", "r_spike", "n_spike"], 
                ["soz_resected","Age_yr", "Male", "r_real", "n_real", "r_pred", "n_pred"]]
    col_list = [["r_real"],["r_spike"],["r_pred"], ["soz_resected","Age_yr", "Male"], 
                ["soz_resected","Age_yr", "Male", "r_real"], 
                ["soz_resected","Age_yr", "Male", "r_spike",], 
                ["soz_resected","Age_yr", "Male", "r_pred", ]]
    
    col_list = [["r_hfo"], ["r_real"],["r_spike"],["r_pred"],
                ["soz_resected"], 
                ["Age_yr", "Male", "r_pred"],
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
    
    auc_list, acc_list = [], []
    for cols in col_list:
        auc, p_val, acc = auc_score(df, cols, folder, plot=plot)
        auc_list.append(auc), acc_list.append(acc)
    return np.array(auc_list), np.array(acc_list), df

def patient_filter(folder, save_suffix, sampling="GMM"):
    threshold = 0.5
    agg_res = aggregate(folder, save_suffix,  sampling=sampling, threshold=threshold)
    #print(agg_res.columns)
    for reconstruct_threshold in [np.inf]:
        res = patient_agg(agg_res, reconstruct_threshold=reconstruct_threshold)
        res = pd.concat(res).sort_values("pt_name")
        df = pd.read_csv("data/meta.csv")
        df = df.merge(res, on="pt_name")
        df["max_HFO_rate"] = df["max_hfo_channel"] / df["length"] * 60 
        save_folder = os.path.join(folder, save_suffix)
        os.makedirs(save_folder, exist_ok=True)
        df.to_csv(f"{save_folder}/auc.csv", index=False)
        df = df[df["resection-status"] ==1]
        #df = df[df["n_hfo"] > 50]
        auc_list, acc_list, df = get_auc_acc(df, save_folder, plot=True)
        #print(reconstruct_threshold, auc_list, acc_list)
if __name__ == "__main__": 
    suffix = sys.argv[1]
    sampling = sys.argv[2]
    save_suffix = sys.argv[3]
    folder = f"res/{suffix}/"
    patient_filter(folder, save_suffix, sampling=sampling)
    print("overall_done")
    #thresold_filter(folder, sampling=sampling)


