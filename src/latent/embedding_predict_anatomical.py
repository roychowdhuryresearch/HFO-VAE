import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fit_embedding import sample
from load_embedding import create_df
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
# random forest
from sklearn.ensemble import RandomForestClassifier
# logistic regression
from sklearn.linear_model import LogisticRegression
import os
from sklearn.metrics import confusion_matrix
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
    # concatenate the train and test
    # df = pd.concat([df, df_test])
    # mu = np.concatenate([mu, mu_test])
    df["mu_index"] = np.arange(len(df))
    df_test["mu_index"] = np.arange(len(df_test))
    #df_info = pd.read_csv(f"./fig_code/figure3_{fold_num}.csv")
    path = f"./res/{name}/fold_{fold_num}/{suffix}/"
    df_info_train = create_df(path, "train_overall_.npz", "train_embedding.csv")
    df_info_test = create_df(path, "test_overall_.npz", "test_embedding.csv")
    df_info = pd.concat([df_info_train, df_info_test])
    df_info = df_info[['pt_names', 'channel_name', 'start', 'end', 'detector','SOZ', 'Resection', 'bad',
       'anatomy', 'Side', 'dataset', 'pathology', 'Age_yr', 'Male',
       'seizure-free', 'resection-status', 'vae-predict', 'real_label', "artifact", "spike", "removed"]]
    df_train = df.merge(df_info, on=["pt_names", "channel_name", "start","end", "detector"])
    df_test = df_test.merge(df_info, on=["pt_names", "channel_name", "start","end", "detector"])
    return df_train, mu, df_test, mu_test

def create_train_data(df, mu, col):
    df = df.dropna(subset=[col])
    df = df[df["vae-predict"] != "Artifact"]
    enum = np.unique(df[col])
    mu_list, label_list, df_list = [], [], []
    for i, a in enumerate(enum):
        index = df[df[col] == a].mu_index
        mu_list.append(mu[index])
        label_list.append(np.ones(len(index))*i)
        df_list.append(df[df[col] == a].copy())
    mu_list = np.concatenate(mu_list)
    label_list = np.concatenate(label_list)
    df = pd.concat(df_list)
    return mu_list, label_list, df

def train_model(mu_train, label_train, mu_test, label_test, df_test = None, save = False):
    clf = RandomForestClassifier(random_state=0, n_jobs=40)
    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(mu_train, label_train)
    pred = clf.predict(mu_test)
    acc = accuracy_score(label_test, pred)
    confusion = confusion_matrix(label_test, pred)
    train_acc = accuracy_score(label_train, clf.predict(mu_train))
    train_confusion = confusion_matrix(label_train, clf.predict(mu_train))
    return acc, confusion, train_acc, train_confusion, pred

def sample_data(df, col, n = 1000):
    # sample n data from each class
    enum = np.unique(df[col])
    df_list = []
    for i, a in enumerate(enum):
        index = np.where(df[col] == a)[0]
        index = np.random.permutation(index)[:n]
        df_list.append(df.iloc[index])
    df = pd.concat(df_list)
    return df

def save_pred(df, save_folder, col, i):
    out_folder = os.path.join(save_folder, "RF_pred", col)
    os.makedirs(out_folder, exist_ok=True)
    df.to_csv(f"{out_folder}/{i}_pred.csv")

def filter_df(df, col, soz = True):
    df = df.dropna(subset=[col])
    df = df[df[col] != "NaN"]
    df = df[df["vae-predict"] != "Artifact"]
    #df = df[df["artifact"] > 0.5]
    #df = df[df["vae-predict"] != "Pathological"]
    if soz:
        #df = df[df["vae-predict"] == "Pathological"]
        df = df[df["SOZ"] == 1]
    else:
        df = df[df["seizure-free"] == 1]
        df = df[df["removed"] == 0]
        #df = df[df["vae-predict"] == "Physiological"]
    return df

def compare_stats(df, mu, col, k = 10):
    # make a copy
    df = [df[0].copy(), df[1].copy()]
    if col == "Age_yr":
        age_ranges = [0,6,11,16,21,float("inf")]
        df[0]["Age_yr"] = df[0]["Age_yr"].apply(lambda x: np.argmax(np.array([x>=age_ranges[i] and x<age_ranges[i+1] for i in range(len(age_ranges)-1)])))
        df[1]["Age_yr"] = df[1]["Age_yr"].apply(lambda x: np.argmax(np.array([x>=age_ranges[i] and x<age_ranges[i+1] for i in range(len(age_ranges)-1)])))
    df_train, df_test = df[0].copy(), df[1].copy()
    mu_train, mu_test = mu[0].copy(), mu[1].copy()
    #df_train, _ = sample(df_train, None, n_sample_df = 500)
    #df_test, _ = sample(df_test, None, n_sample_df = 500)
    df_train = sample_data(df_train, col, n =500)
    df_test = sample_data(df_test, col, n = 500)
    
    feature_train, label_train, df_train = create_train_data(df_train, mu_train, col)
    feature_test, label_test, df_test = create_train_data(df_test, mu_test, col)
    acc_real, confusion_real, train_acc_real, train_confusion_real, pred = train_model(feature_train, label_train, feature_test, label_test)
    df_test["RF_pred"] = pred
    acc_random, confusion_random, train_acc_random, train_confusion_random = [], [], [], []
    for i in range(k):
        label_train = np.random.permutation(label_train)
        label_test = np.random.permutation(label_test)
        acc, confusion, train_acc, train_confusion, _ = train_model(feature_train, label_train, feature_test, label_test)
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
        "df_test": df_test,
    }

def plot_compare_stats(df, mu, col, k = 5, trail = 10, out_folder = "fig/compare_stats", soz = True):
    df = [filter_df(df[0], col, soz), filter_df(df[1], col, soz)]
    res = []
    for i in range(trail):
        r = compare_stats(df, mu, col, k = k)
        res.append(r)
    acc_real = [r["acc_real"] for r in res]
    confusion_real = [r["confusion_real"] for r in res]
    acc_random = [r["acc_random"] for r in res]
    confusion_random = [r["confusion_random"] for r in res]
    train_acc_real = [r["train_acc_real"] for r in res]
    train_confusion_real = [r["train_confusion_real"] for r in res]
    train_acc_random = [r["train_acc_random"] for r in res]
    train_confusion_random = [r["train_confusion_random"] for r in res] 
    df_test = [r["df_test"] for r in res]
    for i, df_ in enumerate(df_test):
        save_pred(df_, out_folder, col, i)
    acc_real = np.concatenate(acc_real)
    acc_random = np.concatenate(acc_random)
    confusion_real = np.array(confusion_real)
    confusion_random = np.concatenate(confusion_random)
    train_acc_real = np.concatenate(train_acc_real)
    train_confusion_real = np.array(train_confusion_real)
    train_acc_random = np.concatenate(train_acc_random)
    train_confusion_random = np.concatenate(train_confusion_random)
    print(acc_real)
    print(acc_random)
    # save acc_real and acc_random
    np.savez(f"{out_folder}/{col}.npz", acc_real = acc_real, acc_random = acc_random)
    fig, ax = plt.subplots(1,6, figsize=(30, 5))
    import seaborn as sns
    sns.swarmplot(data = [acc_random, acc_real], ax = ax[0])
    #ax[0].set_xticklabels(["random", "real"])
    ax[0].set_title(col)
    ax[0].set_ylabel("accuracy")
    cmf_real = confusion_real.mean(axis=0)
    cmf_real = cmf_real / np.sum(cmf_real, axis=1, keepdims=True)
    sns.heatmap(cmf_real, ax = ax[1], annot=True, fmt=".2f")
    if col != "Age_yr":
        ticks = np.unique(df[0].dropna(subset=[col])[col])
    else:
        ticks = ["0-6", "6-11", "11-16", "16-21", ">21"]
    ax[1].set_xticklabels(ticks, rotation=45)
    ax[1].set_yticklabels(ticks, rotation=45)
    ax[1].set_title("confusion matrix")
    # fmt is int
    cmf_random = confusion_random.mean(axis=0)
    cmf_random = cmf_random / np.sum(cmf_random, axis=1, keepdims=True)
    sns.heatmap(cmf_random, ax = ax[2], annot=True, fmt=".2f")
    ax[2].set_title("confusion matrix random")
    sns.swarmplot(data = [train_acc_random, train_acc_real], ax = ax[3])
    #ax[3].set_xticklabels(["random", "real"])
    ax[3].set_title("train accuracy")
    ax[3].set_ylabel("accuracy")
    sns.heatmap(train_confusion_real.mean(axis=0), ax = ax[4], annot=True, fmt=".0f")
    ax[4].set_title("train confusion matrix")
    sns.heatmap(train_confusion_random.mean(axis=0), ax = ax[5], annot=True, fmt=".0f")
    ax[5].set_title("train confusion matrix random")
    os.makedirs(out_folder, exist_ok=True)
    fig.savefig(f"{out_folder}/{col}.png")
    plt.close()

def plot_compare_stats_all(df, mu, k = 1, trail = 10, out_folder = "fig/compare_stats",soz = True):
    col = ["anatomy"]
    for c in col:
        plot_compare_stats(df, mu, c, k = k, trail = trail, out_folder = out_folder, soz = soz)
        
def run(i, pos):
    df_train, mu_train, df_test, mu_test = create_embedding_df(i)
    if pos:
        out_folder = f"fig/compare_stats_soz/fold_{i}"
    else:
        out_folder = f"fig/compare_stats_preseve/fold_{i}"
    plot_compare_stats_all([df_train, df_test], [mu_train, mu_test], k = 2, trail = 5, out_folder = out_folder, soz = pos)

def plot_compare_stats_all_fold():
    import multiprocessing as mp
    params = [(i, pos) for i in range(5) for pos in [True, False]]
    with mp.Pool(10) as p:
        p.starmap(run, params)
    # for p in params:
    #     print(p)
    #     run(*p)
if __name__ == "__main__":
    name = "2023-12-28_1"
    suffix = "10000_2000_81"
    plot_compare_stats_all_fold()
