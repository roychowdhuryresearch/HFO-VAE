import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# import tsne 
#from sklearn.manifold import TSNE
from multiprocessing import Pool
import glob
import seaborn as sns

from sklearn.decomposition import PCA
#from tsnecuda import TSNE
#import umap
import sys
import pickle
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from multiprocessing import Pool

def load_df(fn):
    loaded = np.load(fn, allow_pickle=True)
    df = pd.DataFrame(loaded["labels"], columns=["remove_label", "soz", "removed", "artifact", "spike"])
    df["pt_names"] = loaded["pt_names"]
    df["channel_name"] = loaded["channel_names"]
    df["start"] = loaded["starts"]
    df["end"] = loaded["ends"]
    df["detector"] = loaded["detector"]
    df["pred"] = loaded["pred_with_artifact"]
    df["index"] = np.arange(len(df))
    mu = loaded["mu"]
    return df, mu

def sample(df,mu,n_sample_df = 4000):
    # sample same number for each patient or the max number of samples
    pt_names = df["pt_names"].unique()
    # print(pt_names)
    df = df[df["pt_names"].isin(pt_names)]
    # sample same number for each patient or the max number of samples
    df = df.groupby("pt_names").apply(lambda x: x.sample(n=min(n_sample_df, len(x)), replace=False))
    if mu is None:
        return df, None
    mu = mu[df["index"], :]
    return df, mu


def fit_embedding(train_fn,test_fn, save_fn, n_sample_df = 4000):
    print("train:")
    df_train, mu_train = load_df(train_fn)
    print("test:")
    df_test, mu_test = load_df(test_fn)
    if mu_train.shape[1] > 2:
        from cuml.manifold.t_sne import TSNE
        embedder = TSNE(n_components=2, perplexity=15, n_neighbors=200, verbose=True)
        embedding = embedder.fit_transform(np.concatenate([mu_train, mu_test]))
        mu_train_embed = embedding[:len(mu_train)]
        mu_test_embed = embedding[len(mu_train):]
        #save the embedding
        with open(save_fn+"/embedding.pkl", "wb") as f:
            pickle.dump(embedding, f)
    else:
        mu_train_embed = mu_train
        mu_test_embed = mu_test
    #mu_embed = tsne.fit(mu).transform(mu)
    # tsne = TSNE()
    # mu_train_embed = tsne.fit_transform(mu_train)
    # mu_test_embed = tsne.fit_transform(mu_test)

    df_train["mu_embed_0"] = mu_train_embed[:, 0]
    df_train["mu_embed_1"] = mu_train_embed[:, 1]

    df_test["mu_embed_0"] = mu_test_embed[:, 0]
    df_test["mu_embed_1"] = mu_test_embed[:, 1]

    df_train.to_csv(save_fn+"/train_embedding.csv")
    df_test.to_csv(save_fn+"/test_embedding.csv")
    

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    date = sys.argv[1]
    save_suffix = sys.argv[2]
    sampling = "GMM"
    result_dir = f"res/{date}"
    paths = glob.glob(f"{result_dir}/fold_*/{save_suffix}")
    params = [[path+"/train_overall_.npz",path+"/test_overall_.npz", path] for path in paths if os.path.exists(path+"/test_overall_.npz")]
    with Pool(5) as p:
        p.starmap(fit_embedding, params)