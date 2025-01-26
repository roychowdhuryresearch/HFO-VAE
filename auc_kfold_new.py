from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
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
import src.param as param
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

    df_pp = pd.DataFrame({"pt_name": [pt_name], "r_hfo":[len(df_p[df_p["removed"] == 1])/len(df_p)], "n_hfo": [len(df_p)],"r_real": [r_artifact], 
                        "r_spike": [r_spike], "r_pred": [r_pred], 
                        "n_real": [n_artifact], "n_spike": [n_spike], "n_pred": [n_pred],
                        "n_pred_agree_spike": [n_pred_agree_spike], "n_spike_agree_pred": [n_spike_agree_pred], "soz_resected":[soz_resected], 
                        "fold": [fold_num]
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
    df["fold"] = fn.split("/")[-2].split("_")[-1]
    return df

def aggregate(folder, save_surffix ,fold, threshold = 0.5,sampling="GMM"):
    if sampling == "GMM":
        test_fn = f"{folder}/fold_{fold}/{save_surffix}/test_overall_.npz"
        train_fn = f"{folder}/fold_{fold}/{save_surffix}/train_overall_.npz"
    else:
        test_fn = f"{folder}/fold_{fold}/test_kmeans_new.npz"
        train_fn = f"{folder}/fold_{fold}/train_kmeans_new.npz"

    return process_fn(test_fn, threshold), process_fn(train_fn, threshold)

def get_pval(X, y):
    mod = sm.Logit(y,np.squeeze(X))
    fii = mod.fit(disp=False)
    p_values = fii.summary2().tables[1]['P>|z|'].to_dict()
    p_values = {k: round(v, 3) for k, v in p_values.items()}
    return p_values
# import random forest
from sklearn.ensemble import RandomForestClassifier   
from sklearn.svm import SVC
from sklearn.model_selection import KFold


def split_kfolds(df, folds=5):
    datasets_splits = []
    for i in np.unique(df["fold"].values):
        datasets_splits.append([df[df["fold"] != i], df[df["fold"] == i]])

    return datasets_splits
    
def auc_score(df_train, df_test, cols, save_path, fold = 0):
    save_path_use = save_path + "/" + str(cols)
    os.makedirs(save_path_use, exist_ok=True)

    # ground_truths, pred_probs, preds = [], [], []
    clf = LogisticRegression(random_state=0, max_iter=1000, class_weight="balanced", penalty="l2")
    clf.fit(df_train[cols], df_train["seizure-free"])
    ground_truths = df_train["seizure-free"]
    pred_probs = clf.predict_proba(df_train[cols])[:,1]
    preds = clf.predict(df_train[cols])
    plot_auc(ground_truths, pred_probs, preds, f"{save_path_use}/{fold}_train.png", cols)
    ground_truths = df_test["seizure-free"]
    pred_probs = clf.predict_proba(df_test[cols])[:,1]
    preds = clf.predict(df_test[cols])
    plot_auc(ground_truths, pred_probs, preds, f"{save_path_use}/{fold}_test.png", cols)


    # clf.fit(df[cols], df["seizure-free"])
    return ground_truths, preds, pred_probs

def plot_auc(ground_truth, pred_prob, preds,save_path,cols):
    auc = roc_auc_score(ground_truth, pred_prob)
    fpr, tpr, _ = metrics.roc_curve(ground_truth,  pred_prob)
    acc = metrics.accuracy_score(ground_truth,  preds)
    # if plot:
    auc_str = "{0:.3f}".format(auc)
    plt.figure(figsize=(4.2,4.2))
    plt.plot([0, 1], [0, 1],'o--')
    plt.plot(fpr,tpr,label=f"AUC="+auc_str)
    #plt.plot(fpr,tpr,label=f"AUC="+str(auc) + "\npval=" + str(p_val))
    plt.title(f"ROC curve {cols} \n acc={acc}, fold = {save_path.split('/')[-1][:-4]}")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig(save_path)
    # plt.savefig(f"{save_path}_{cols}_roc.png")
    plt.close()
    # return auc, acc

def get_auc_acc(dfs,save_path):
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
                ["Age_yr", "Male"],
                ["soz_resected","r_hfo"], 
                ["soz_resected","r_real"], 
                ["soz_resected","r_spike",], 
                ["soz_resected","r_pred",],
                ["soz_resected","Age_yr", "Male","r_hfo"], 
                ["soz_resected","Age_yr", "Male","r_real"], 
                ["soz_resected","Age_yr", "Male","r_spike",], 
                ["soz_resected","Age_yr", "Male","r_pred",],
                ]
    
    # auc_list, acc_list = [], []
    for cols in col_list:
        ground_truth, pred, pred_prob = [],[],[]
        for key in dfs.keys():
            test_df, train_df = dfs[key]
            g, p, p_prob = auc_score(train_df, test_df, cols, save_path, fold = key)
            ground_truth.append(g)
            pred.append(p)
            pred_prob.append(p_prob)
        ground_truth = np.concatenate(ground_truth)
        pred = np.concatenate(pred)
        pred_prob = np.concatenate(pred_prob)
        plot_auc(ground_truth, pred_prob, pred, f"{save_path}/{cols}/all.png", cols)
        # auc_list.append(auc), acc_list.append(acc)
    # return np.array(auc_list), np.array(acc_list), df

def setup_df(agg_res, meta_df, reconstruct_threshold, n_threshold = 30):
        
        res = patient_agg(agg_res, reconstruct_threshold=reconstruct_threshold)
        res = pd.concat(res).sort_values("pt_name")
        # df = pd.read_csv("data/meta.csv")

        df = meta_df.merge(res, on="pt_name")
        #df = df[~df["pt_name"].isin(new_patents)]
        # df.to_csv(f"{folder}/auc.csv", index=False)
        df = df[df["resection-status"] ==1]
        #df = df[df["n_hfo"] > n_threshold]
        return df

def patient_filter(folder, save_surffix, sampling="GMM"):
    threshold = 0.5
    meta = pd.read_csv(param.get_args()["meta_fn"])
           # if os.path.exists(f"{folder}/{save_surffix}/auc_kfold_{n_threshold}.xlsx"):
        #     continue
    dfs = {}
    auc_kfold_df = []
    train_list_dict , test_list_dict = {}, {}
    for fold in range(len(glob.glob(f"{folder}/fold_*"))):
        test_df, train_df = aggregate(folder, save_surffix, fold, sampling=sampling, threshold=threshold)
        auc_kfold_df.append(test_df)
        test_df = setup_df(test_df, meta, reconstruct_threshold=np.inf)
        train_df = setup_df(train_df, meta, reconstruct_threshold=np.inf)
        # print("test_df.shape", test_df.shape)
        # print("train_df.shape", train_df.shape)
        dfs[fold] = [test_df, train_df]
        train_list_dict[fold] = train_df
        test_list_dict[fold] = test_df

    for n_threshold in np.arange(0,60,5):
        dfs_filter = {}
        for key in dfs.keys():
            test_df = dfs[key][0]
            train_df = dfs[key][1]
            test_df = test_df[test_df["n_hfo"] > n_threshold].copy()
            train_df = train_df[train_df["n_hfo"] > n_threshold].copy()
            dfs_filter[key] = [test_df, train_df]
        auc_kfold_df_filter = []
        for key in dfs_filter.keys():
            test_df, train_df = dfs_filter[key]
            auc_kfold_df_filter.append(test_df)
        test_list_dict_filter = {}
        train_list_dict_filter = {}
        for key in dfs_filter.keys():
            test_list_dict_filter[key] = dfs_filter[key][0]
            train_list_dict_filter[key] = dfs_filter[key][1]
        save_folder = os.path.join(folder, save_surffix)
        #get_auc_acc(dfs, save_folder+"/ROC_plots")
        auc_kfold_df = pd.concat(auc_kfold_df_filter)
        #auc_kfold_df.to_csv(f"{save_folder}/auc_kfold.csv", index=False)
        # to excel
        writer = pd.ExcelWriter(f"{save_folder}/auc_kfold_{n_threshold}.xlsx", engine='xlsxwriter')
        for fold in range(len(glob.glob(f"{folder}/fold_*"))):
            train_list_dict_filter[fold].to_excel(writer, sheet_name=f"train_{fold}")
            test_list_dict_filter[fold].to_excel(writer, sheet_name=f"test_{fold}")
        writer.close()

if __name__ == "__main__": 
    surffix = "2023-12-28_2"
    sampling = "GMM"
    save_surfix = "10000_2000_81"

    # surffix = "2023-11-02_21"
    # save_surfix = "10000_2000_81"
    # for save_surfix in ["10000_2000_81", "10000_1000_81", "5000_1000_81", "5000_2000_81"]:
    folder = f"res/{surffix}"
    patient_filter(folder, save_surfix, sampling=sampling)
        #thresold_filter(folder, sampling=sampling)


