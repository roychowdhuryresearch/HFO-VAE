import os
import numpy as np
import pandas as pd
import pickle
import argparse

from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_score, recall_score, f1_score

###############################################################################
# Utility Functions
###############################################################################

def eval_metrics(labels, preds):
    """
    Compute classification metrics given true labels and predictions.
    labels, preds are 1D arrays of 0/1.
    Returns {'acc', 'recall', 'precision', 'f1'}.
    """
    acc = np.mean(preds == labels)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"acc": acc, "recall": recall, "precision": precision, "f1": f1}

def load_npz(fn):
    """
    Load data from an NPZ file.
    Returns (df, mu_array).
    """
    data = np.load(fn)
    df = pd.DataFrame({
        "pt_names": data['pt_names'],
        "channel_names": data['channel_names'],
        "starts": data['starts'],
        "ends": data['ends'],
        "detector": data['detector'],
        "spike": data['labels'][:, -1],    # last column
        "real": data['labels'][:, -2],     # second-to-last column
        "remove": data['labels'][:, -3],   # third-to-last column
        "reconstruct": data['reconstruct']
    })
    mu = data["mu"]
    return df, mu

def sample_uniform(df, mu, num_samples=10000, seed=42):
    """
    Randomly sample up to `num_samples` rows per patient.
    Returns (df_sampled, mu_sampled).
    """
    np.random.seed(seed)
    idxs = []
    for pt in df["pt_names"].unique():
        pt_indices = df[df["pt_names"] == pt].index
        sample_size = min(num_samples, len(pt_indices))
        idxs.append(np.random.choice(pt_indices, size=sample_size, replace=False))
    idxs = np.concatenate(idxs)
    return df.iloc[idxs].copy(), mu[idxs].copy()

###############################################################################
# Artifact Rejection: GMM
###############################################################################

def fit_gmm_artifact(df, mu, num_samples=10000, seed=42):
    """
    Fit a GMM on [mu, reconstruct] to classify artifact vs. non-artifact.
    The cluster with the largest mean reconstruction is labeled 'artifact_class'.
    Returns { "gmm": fitted_gmm, "artifact_class": int }.
    """
    # Sample for efficiency
    df_s, mu_s = sample_uniform(df, mu, num_samples=num_samples, seed=seed)
    feats_s = np.concatenate([mu_s, df_s["reconstruct"].values.reshape(-1, 1)], axis=1)

    # Fit 2-component GMM
    gmm_model = GaussianMixture(n_components=2, random_state=seed, max_iter=2000, tol=1e-6)
    gmm_model.fit(feats_s)

    # Predict on full data
    feats_full = np.concatenate([mu, df["reconstruct"].values.reshape(-1, 1)], axis=1)
    preds_full = gmm_model.predict(feats_full)

    recons = df["reconstruct"].values
    mean0 = recons[preds_full == 0].mean() if np.sum(preds_full == 0) else float('-inf')
    mean1 = recons[preds_full == 1].mean() if np.sum(preds_full == 1) else float('-inf')

    artifact_class = 0 if mean0 > mean1 else 1

    return {"gmm": gmm_model, "artifact_class": artifact_class}

def predict_gmm_artifact(df, mu, gmm_artifact_dict):
    """
    Returns a boolean mask: True => non-artifact, False => artifact.
    """
    gmm_model = gmm_artifact_dict["gmm"]
    artifact_class = gmm_artifact_dict["artifact_class"]

    feats = np.concatenate([mu, df["reconstruct"].values.reshape(-1,1)], axis=1)
    preds = gmm_model.predict(feats)
    return (preds != artifact_class)  # True => NOT artifact

def eval_gmm_artifact(df, mu, labels_artifact, gmm_artifact_dict):
    """
    Evaluate GMM-based artifact detection. labels_artifact=1 => artifact.
    Because predict_gmm_artifact => True if non-artifact, we invert for compare.
    """
    preds = ~predict_gmm_artifact(df, mu, gmm_artifact_dict)
    return eval_metrics(labels_artifact, preds)

###############################################################################
# Pathologic Spike Detection (GMM)
###############################################################################

def fit_gmm_pathology(df, mu, num_samples=1000, seed=42):
    """
    Fit GMM on [mu, reconstruct] for pathologic spike detection.
    The cluster with the highest ratio of remove==1 in resected pts is 'pathologic_class'.
    Returns { "gmm": fitted_gmm, "pathologic_class": int }.
    """
    feats_all = np.concatenate([mu, df["reconstruct"].values.reshape(-1,1)], axis=1)

    # Sample
    _, feats_s = sample_uniform(df, feats_all, num_samples=num_samples, seed=seed)

    gmm_model = GaussianMixture(n_components=2, random_state=seed, max_iter=2000, tol=1e-6)
    gmm_model.fit(feats_s)

    path_class = _identify_pathologic_cluster(df, feats_all, gmm_model)
    return {"gmm": gmm_model, "pathologic_class": path_class}

def _identify_pathologic_cluster(df, feats, model):
    """
    Among 2 GMM clusters, pick cluster with highest ratio of remove==1 in resected patients.
    """
    # Resection-based patients
    resected_pts = []
    for pt in df["pt_names"].unique():
        df_pt = df[df["pt_names"] == pt]
        if df_pt["remove"].sum() > 0 and len(df_pt) > 50:
            resected_pts.append(pt)

    mask = df["pt_names"].isin(resected_pts)
    preds = model.predict(feats[mask])
    resected_df = df[mask].copy()
    resected_df["pred"] = preds

    # Ratios of remove==1 in each cluster
    ratios = []
    for c in [0,1]:
        denom = np.sum(resected_df["pred"] == c)
        numer = np.sum((resected_df["pred"] == c) & (resected_df["remove"] == 1))
        ratio = numer / denom if denom > 0 else 0
        ratios.append(ratio)

    return np.argmax(ratios)

def predict_gmm_pathology(df, mu, gmm_path_dict):
    """
    Returns a boolean mask: True => pathologic spike, False => not pathologic.
    """
    feats = np.concatenate([mu, df["reconstruct"].values.reshape(-1,1)], axis=1)
    preds = gmm_path_dict["gmm"].predict(feats)
    return (preds == gmm_path_dict["pathologic_class"])

def eval_gmm_pathology(df, mu, labels_spike, gmm_path_dict):
    """
    Evaluate pathologic spike detection. labels_spike=1 => pathologic spike.
    """
    preds = predict_gmm_pathology(df, mu, gmm_path_dict)
    return eval_metrics(labels_spike, preds)

###############################################################################
# Pipeline (GMM-only) for Train/Test
###############################################################################

def run_pipeline_gmm(train_path,
                     test_path,
                     artifact_num_samples=10000,
                     pathology_num_samples=1000,
                     seed=42):
    """
    1) Load train/test data from user-specified paths
    2) Fit GMM artifact rejection on train
    3) Filter out artifact from train
    4) Fit GMM pathology on "kept" train data
    5) Return necessary dictionaries + data
    """
    train_df, mu_train = load_npz(train_path)
    test_df, mu_test = load_npz(test_path)

    # Fit artifact GMM on train
    artifact_fit = fit_gmm_artifact(train_df, mu_train, artifact_num_samples, seed=seed)
    print("eval_gmm_artifact", eval_gmm_artifact(train_df, mu_train, train_df["real"] < 0.5, artifact_fit))
    artifact_model = {
        "gmm": artifact_fit["gmm"],
        "artifact_class": artifact_fit["artifact_class"]
    }

    # Keep only non-artifact in train
    keep_mask = predict_gmm_artifact(train_df, mu_train, artifact_model)
    kept_df = train_df[keep_mask].reset_index(drop=True)
    kept_mu = mu_train[keep_mask]

    # Fit pathologic GMM on kept data
    path_fit = fit_gmm_pathology(kept_df, kept_mu, pathology_num_samples, seed=seed)
    print("eval_gmm_pathology", eval_gmm_pathology(kept_df, kept_mu, kept_df["spike"] > 0.5, path_fit))

    path_model = {
        "gmm": path_fit["gmm"],
        "pathologic_class": path_fit["pathologic_class"]
    }

    return {
        "train_df": train_df,
        "mu_train": mu_train,
        "test_df": test_df,
        "mu_test": mu_test,
        "artifact_model": artifact_model,
        "path_model": path_model
    }

def predict_pipeline(df, mu, artifact_model, path_model, with_artifact=False):
    """
    1) Predict artifact => True if non-artifact
    2) Predict pathologic => True if pathologic
    Returns:
      - if with_artifact=False => boolean array (pathologic & not artifact)
      - if with_artifact=True  => array of -1 (artifact), 0 (neither), +1 (pathologic)
    """
    # Non-artifact
    artifact_mask = predict_gmm_artifact(df, mu, artifact_model)

    # Pathologic
    path_mask = predict_gmm_pathology(df, mu, path_model)

    if with_artifact:
        # -1 => artifact, +1 => pathologic, 0 => neither
        return path_mask.astype(int) - (~artifact_mask).astype(int)
    else:
        return artifact_mask & path_mask

def save_predictions(train_file, test_file, artifact_model, path_model, save_suffix="results"):
    """
    Predict on train/test. Save each as [save_dir]/(train|test)_overall_.npz.
    Skips large arrays 'in'/'out' in the original NPZ for memory reasons.
    """
    base_dir = os.path.dirname(train_file)
    save_dir = os.path.join(base_dir, save_suffix)
    os.makedirs(save_dir, exist_ok=True)

    # --- TRAIN ---
    data_train = np.load(train_file)
    df_train, mu_train = load_npz(train_file)

    preds_train = predict_pipeline(df_train, mu_train, artifact_model, path_model, with_artifact=False)
    preds_train_art = predict_pipeline(df_train, mu_train, artifact_model, path_model, with_artifact=True)

    data_train_out = {k: data_train[k] for k in data_train.keys() if k not in ["in", "out"]}
    data_train_out["pred"] = preds_train
    data_train_out["pred_with_artifact"] = preds_train_art
    data_train_out["pred_prob"] = np.zeros(preds_train.shape, dtype=float)

    np.savez(os.path.join(save_dir, "train"), **data_train_out)

    # --- TEST ---
    data_test = np.load(test_file)
    df_test, mu_test = load_npz(test_file)

    preds_test = predict_pipeline(df_test, mu_test, artifact_model, path_model, with_artifact=False)
    preds_test_art = predict_pipeline(df_test, mu_test, artifact_model, path_model, with_artifact=True)

    data_test_out = {k: data_test[k] for k in data_test.keys() if k not in ["in", "out"]}
    data_test_out["pred"] = preds_test
    data_test_out["pred_with_artifact"] = preds_test_art
    data_test_out["pred_prob"] = np.zeros(preds_test.shape, dtype=float)

    np.savez(os.path.join(save_dir, "test"), **data_test_out)

###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    """
    Example usage:
        python script.py --train_file /path/to/train_data.npz \
                         --test_file /path/to/test_data.npz \
                         --artifact_num_samples 10000 \
                         --pathology_num_samples 3000 \
                         --seed 42
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="/mnt/SSD5/yipeng/VAE/res/2023-12-28_3/fold_0/train_81.npz",
                        help="Path to the 'train' NPZ file.")
    parser.add_argument("--test_file", type=str, default="/mnt/SSD5/yipeng/VAE/res/2023-12-28_3/fold_0/test_81.npz",
                        help="Path to the 'test' NPZ file.")
    parser.add_argument("--artifact_num_samples", type=int, default=10000,
                        help="Number of samples for artifact GMM training.")
    parser.add_argument("--pathology_num_samples", type=int, default=3000,
                        help="Number of samples for pathologic GMM training.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--save_suffix", type=str, default="results",
                        help="Folder name for saving predictions.")
    args = parser.parse_args()

    # 1) Run pipeline
    pipeline_dict = run_pipeline_gmm(
        train_path=args.train_file,
        test_path=args.test_file,
        artifact_num_samples=args.artifact_num_samples,
        pathology_num_samples=args.pathology_num_samples,
        seed=args.seed
    )

    # 2) Save predictions
    save_predictions(
        train_file=args.train_file,
        test_file=args.test_file,
        artifact_model=pipeline_dict["artifact_model"],
        path_model=pipeline_dict["path_model"],
        save_suffix=args.save_suffix
    )
