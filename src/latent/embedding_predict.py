import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fit_embedding import sample
from load_embedding import create_df   
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.svm import SVC
# random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.metrics import confusion_matrix
import seaborn as sns
import datetime
# import MANOVA
from statsmodels.multivariate.manova import MANOVA
import glob
def create_df_mu(fn):
    data = np.load(fn)
    mu = data['mu']
    # base on all .files in the folder except for the mu
    df = pd.DataFrame({k: data[k] for k in data.files if k != 'mu' and k != 'labels'})
    df = df[["pt_names", "channel_names", "starts","ends", "detector", "reconstruct"]]
    df.columns = ["pt_names", "channel_name", "start","end", "detector", "reconstruct"]
    return df, mu

def create_embedding_df(fold_num):
    fn = f"./res/{name}/fold_{fold_num}/{suffix}/train_overall_.npz"
    df, mu = create_df_mu(fn)
    fn = f"./res/{name}/fold_{fold_num}/{suffix}/test_overall_.npz"
    df_test, mu_test = create_df_mu(fn)

    # add mu 
    col_names = [f"feature_{i}" for i in range(mu.shape[1])]
    df = pd.concat([df, pd.DataFrame(mu, columns=col_names)], axis=1)
    df_test = pd.concat([df_test, pd.DataFrame(mu_test, columns=col_names)], axis=1)

    path = f"./res/{name}/fold_{fold_num}/{suffix}/"
    df_info_train = create_df(path, "train_overall_.npz", "train_embedding.csv")
    df_info_test = create_df(path, "test_overall_.npz", "test_embedding.csv")
    # df_info_test[df_info_test["pt_names"] == "Pt1_AR"].to_csv("fig/compare_stats/df.csv")

    df_info = pd.concat([df_info_train, df_info_test])
    df_info = df_info[['pt_names', 'channel_name', 'start', 'end', 'detector','SOZ', 'Resection', 'bad',
       'anatomy', 'Side', 'dataset', 'pathology', 'Age_yr', 'Male',
       'seizure-free', 'resection-status', 'vae-predict', 'real_label', "artifact", "spike", "removed"]]
    df_train = df.merge(df_info, on=["pt_names", "channel_name", "start","end", "detector"])
    df_test = df_test.merge(df_info, on=["pt_names", "channel_name", "start","end", "detector"])
    df_train = df_train[(df_train["vae-predict"] != "Artifact") & (df_train["artifact"] > 0.5)]
    df_test = df_test[(df_test["vae-predict"] != "Artifact") & (df_test["artifact"] > 0.5)]
    df_train["fold"] = fold_num
    df_test["fold"] = fold_num
    return df_train, df_test

def create_train_data(df, col):
    fold_num = df["fold"].unique()[0]
    lookup = {0:1, 1:5,2:0, 3:3,4:5}

    df = df.dropna(subset=[col])
    df = df[df["vae-predict"] != "Artifact"]
    enum = np.unique(df[col])
    df_list = []
    for i, a in enumerate(enum):
        df_e = df[df[col] == a].copy()
        df_e["label"] = i
        df_list.append(df_e)
        mu = df_e[[f"feature_{i}" for i in range(8)]].values
    df = pd.concat(df_list)
    return df

def train_model(df_train, df_test, save = False):
    mu_train = df_train[[col for col in df_train.columns if "feature" in col]].values
    mu_test = df_test[[col for col in df_test.columns if "feature" in col]].values
    label_train = df_train["label"].values
    label_test = df_test["label"].values

    clf = LogisticRegression(random_state=0)
    clf.fit(mu_train, label_train)
    pred = clf.predict(mu_test)
    acc = accuracy_score(label_test, pred)
    confusion = confusion_matrix(label_test, pred)
    train_acc = accuracy_score(label_train, clf.predict(mu_train))
    train_confusion = confusion_matrix(label_train, clf.predict(mu_train))
    return acc, confusion, train_acc, train_confusion, pred

def sample_data(df:pd.DataFrame, col, n = 1000, random_state = 0):
    def even_sample(df, n):
        patient_names = df["pt_names"].unique()
        np.random.seed(random_state)
        patient_names = np.random.permutation(patient_names)[:3]
        df = df[df["pt_names"].isin(patient_names)].sample(n=n, replace=True, random_state=random_state)
        return df
    
    # sample n data from each class
    enum = np.unique(df[col])
    df_list = []
    for i, a in enumerate(enum):
        df_e = df[df[col] == a]
        df_e = even_sample(df_e, n)
        df_list.append(df_e)
    df = pd.concat(df_list)
    return df

def save_pred(df, pred, col, i):
    df["RF_pred"] = pred
    out_folder = f"./fig/compare_stats/RF_predict/{col}"
    os.makedirs(out_folder, exist_ok=True)
    df.to_csv(f"{out_folder}/{i}_pred.csv")

def filter_df(df, col, n = 1000):
    n = n//4
    df = df.dropna(subset=[col])
    df = df[df[col] != "NaN"]
    df = df[df["vae-predict"] != "Artifact"]
    if col == "pathology":
        # rename tumor to Tumor
        df[col] = df[col].apply(lambda x: "Tumor" if x == "tumor" else x)
        names = ["HS", "FCD", "Tumor", "Others"]
        # rename pathology
        df[col] = df[col].apply(lambda x: "Others" if x not in names else x)
    return df

def manova(mus, labels):
    mus = np.concatenate(mus)
    labels = np.concatenate(labels)
    df = pd.DataFrame(mus)
    df["label"] = labels
    df.columns = [f"feature_{i}" for i in range(mus.shape[1])] + ["label"]
    manova = MANOVA.from_formula('feature_0 + feature_1 + feature_2 + feature_3 + feature_4 + feature_5 + feature_6 + feature_7 ~ label', data=df)
    result = manova.mv_test()
    print(result.summary())

def shuffle_patient_wise(df, randon_state=0):
    # get patient names and label as dict
    np.random.seed(randon_state)
    pt_names = df["pt_names"].unique()
    pt_name_lookup = {pt_name: df[df["pt_names"] == pt_name]["label"].values[0] for pt_name in pt_names}
    # make it a list
    pt_names = list(pt_names)
    pt_names_label = [pt_name_lookup[pt_name] for pt_name in pt_names]
    # shuffle
    pt_names = np.random.permutation(pt_names)
    res_ = []
    for pt_name, label in zip(pt_names, pt_names_label):
        df_p = df[df["pt_names"] == pt_name].copy()
        df_p["label"] = label
        res_.append(df_p)
    df = pd.concat(res_)
    return df


def compare_stats(df, col, k = 10, trail = 0, seed = 0):
    fold , t = int(trail.split("_")[0]), int(trail.split("_")[1])
    # make a copy
    df = [df[0].copy(), df[1].copy()]
    #print(df[0][col].unique(), df[1][col].unique())
    # if "gliosis " in df[0][col].unique():
    #     print("pt_name :" , df[0][df[0][col] == "gliosis "]["pt_names"].unique())
    if col == "Age_yr":
        age_ranges = [0,6,11,16,21,float("inf")]
        df[0]["Age_yr"] = df[0]["Age_yr"].apply(lambda x: np.argmax(np.array([x>=age_ranges[i] and x<age_ranges[i+1] for i in range(len(age_ranges)-1)])))
        df[1]["Age_yr"] = df[1]["Age_yr"].apply(lambda x: np.argmax(np.array([x>=age_ranges[i] and x<age_ranges[i+1] for i in range(len(age_ranges)-1)])))
    df_train, df_test = df[0].copy(), df[1].copy()
    df_train = sample_data(df_train, col, n =100, random_state=seed)
    df_test = sample_data(df_test, col, n = 100, random_state=seed+1)
    
    df_train = create_train_data(df_train, col)
    df_test = create_train_data(df_test, col)
    #manova([feature_train, feature_test], [label_train, label_test])
    acc_real, confusion_real, train_acc_real, train_confusion_real, pred = train_model(df_train, df_test, save=True)
    save_pred(df_test, pred, col, trail)
    acc_random, confusion_random, train_acc_random, train_confusion_random = [], [], [], []
    for i in range(k):
        #df_train["label"] = df_train["label"].values[np.random.permutation(df_train.shape[0])]
        #df_test["label"] = df_test["label"].values[np.random.permutation(df_test.shape[0])]
        df_train = shuffle_patient_wise(df_train, randon_state=seed)
        df_test = shuffle_patient_wise(df_test, randon_state=seed+1)
        acc, confusion, train_acc, train_confusion, _ = train_model(df_train, df_test)
        acc_random.append(acc)
        confusion_random.append(confusion)
        train_acc_random.append(train_acc)
        train_confusion_random.append(train_confusion)
    acc_random = np.array(acc_random)
    confusion_random = np.array(confusion_random)
    train_acc_random = np.array(train_acc_random)
    train_confusion_random = np.array(train_confusion_random)
    return {
        "acc_real": np.array([acc_real]),
        "confusion_real": confusion_real,
        "train_acc_real": np.array([train_acc_real]),
        "train_confusion_real": train_confusion_real,
        "acc_random": acc_random,
        "confusion_random": confusion_random,
        "train_acc_random": train_acc_random,
        "train_confusion_random": train_confusion_random,
    }


def plot_compare_stats(df, col, k = 5, trail = 10, out_folder = "fig/compare_stats", seed = 0):
    df = [filter_df(df[0].copy(), col, 1000), filter_df(df[1].copy(), col, 1000)]
    res = []
    for i in range(trail):
        fold_num = out_folder.split("/")[-1].split("_")[-1]
        r = compare_stats(df, col, k = k, trail=f"{fold_num}_{i}", seed = seed+i)
        res.append(r)
    acc_real = [r["acc_real"] for r in res]
    confusion_real = [r["confusion_real"] for r in res]
    acc_random = [r["acc_random"] for r in res]
    confusion_random = [r["confusion_random"] for r in res]
    train_acc_real = [r["train_acc_real"] for r in res]
    train_confusion_real = [r["train_confusion_real"] for r in res]
    train_acc_random = [r["train_acc_random"] for r in res]
    train_confusion_random = [r["train_confusion_random"] for r in res] 
    acc_real = np.concatenate(acc_real)
    acc_random = np.concatenate(acc_random)
    confusion_real = np.array(confusion_real)
    confusion_random = np.concatenate(confusion_random)
    train_acc_real = np.concatenate(train_acc_real)
    train_confusion_real = np.array(train_confusion_real)
    train_acc_random = np.concatenate(train_acc_random)
    train_confusion_random = np.concatenate(train_confusion_random)
    # save acc real and random
    os.makedirs(out_folder, exist_ok=True)
    np.savez(f"{out_folder}/{col}_acc_{seed}.npz", acc_real=acc_real, acc_random=acc_random, confusion_real=confusion_real, confusion_random=confusion_random)
   
def plot_compare_stats_all(df, k = 5, trail = 10, out_folder = "fig/compare_stats", seed = 0, col = None):
    plot_compare_stats(df, col, k = k, trail = trail, out_folder = out_folder, seed = seed)

def run(i, col, seed):
    df_train, df_test = create_embedding_df(i)
    out_folder = f"fig/compare_stats/fold_{i}"
    plot_compare_stats_all([df_train, df_test], k = 1, trail = 5, out_folder = out_folder, seed = seed, col = col)

def plot_compare_stats_all_fold():
    from multiprocessing import Pool
    col = ["dataset", "Age_yr", "Male", "pathology"]
    seed = np.arange(10)
    fold = np.arange(5)
    params = [(i, c, s) for i in fold for c in col for s in seed]
    with Pool(40) as p:
        p.starmap(run, params)

    def process_acc(column, s):
        acc, acc_random = [], []
        fns = glob.glob(f'./fig/compare_stats/fold_*/{column}_acc_{s}.npz') # These files are generated by fig_code/embedding_pred.py
        for fn in fns:
            data = np.load(fn)
            acc.append(data['acc_real'])
            acc_random.append(data['acc_random'])
        acc = np.array(acc).reshape(-1)
        acc_random = np.array(acc_random).reshape(-1)
        # t-test
        from scipy.stats import ttest_ind
        p_val_ttest = ttest_ind(acc, acc_random, alternative='greater')[1]
        return p_val_ttest, acc.mean(), acc_random.mean()
    for s in seed:
        for c in col:
            p_val, acc, acc_random = process_acc(c, s)
            print(s, c, f"p_val: {p_val:.2f}, acc: {acc:.2f}, acc_random: {acc_random:.2f}")

if __name__ == "__main__":
    name = "2023-12-28_1"
    suffix = "10000_2000_81"
    plot_compare_stats_all_fold()

