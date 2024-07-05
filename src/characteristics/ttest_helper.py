import pandas as pd
import numpy as np
import os
from scipy import stats
from multiprocessing import Pool
from functools import partial

def load_patient(data_folder, pt_name):
    if pt_name.startswith("M"):
        ins = "detroit"
    elif pt_name.startswith("SEEG"):
        ins = "seeg"
    else:
        ins = "ucla"
    patient_data = np.load(os.path.join(data_folder, ins ,pt_name,"feature.npz"), allow_pickle=True)
    starts = patient_data["start"]
    ends = patient_data["end"]
    channel_names = patient_data["channel_name"]
    pt_df = pd.DataFrame(
        {
            "start": starts,
            "end": ends,
            "channel_name": channel_names,
        }
    )
    pt_df["index"] = np.arange(len(starts))
    pt_df["pt_names"] = pt_name
    data = patient_data["feature"]
    #data = (data - np.min(data, axis=(1, 2), keepdims=True)) / (np.max(data, axis=(1, 2), keepdims=True) - np.min(data, axis=(1, 2), keepdims=True))
    # normalize to 0-1
    #data = data.reshape(data.shape[0], -1)
    #data = (data - np.min(data, axis=1, keepdims=True)) / (np.max(data, axis=1, keepdims=True) - np.min(data, axis=1, keepdims=True))
    data = data.reshape(-1, 64, 64)
    return pt_df, data

def ttest(data, pos_index, neg_index):
    if pos_index.shape[0] == 0 or neg_index.shape[0] == 0:
        return np.zeros(data[0].shape) -1
    pos_data = data[pos_index]
    neg_data = data[neg_index]
    pos_data = pos_data.reshape(pos_data.shape[0], -1)
    neg_data = neg_data.reshape(neg_data.shape[0], -1)
    try:
        ttest = stats.ttest_ind(pos_data, neg_data, axis=0, equal_var=False, alternative="greater")[1]
        ttest = ttest.reshape(data[0].shape)
    except:
        return np.zeros(data[0].shape) -1
    return (ttest < 0.05).astype(int)

def ttest_one_patent(data_folder, pt_name, col, df):
    pt_df, data = load_patient(data_folder, pt_name)
    pt_df = pt_df[["pt_names", "channel_name", "start", "end", "index"]]
    df.drop(columns=["index"], inplace=True)
    pt_df = pt_df.merge(df, on=["pt_names", "channel_name", "start", "end"])
    pos_index = pt_df[(pt_df[col] == 1)]["index"].values
    neg_index = pt_df[pt_df[col] == 0]["index"].values
    return ttest(data, pos_index, neg_index)

def ttest_patients(df, col):
    pt_names = df["pt_names"].unique()
    data_folder = "/mnt/SSD5/lawrence/VAE/data"
    params = [(data_folder, pt_name, col, df[df["pt_names"] == pt_name]) for pt_name in pt_names]
    with Pool(32) as p:
        ttest_result = p.starmap(ttest_one_patent, params)
    res = []
    for r in ttest_result:
        if type(r) == tuple:
            res.append(r[0])
        else:
            res.append(r)
    clean_res = []
    for r in res:
        if np.sum(r == -1) > 0:
            print("error")
        else:
            clean_res.append(r)
    return clean_res

def retrive_one_patient(data_folder, pt_name, col, df):
    pt_df, data = load_patient(data_folder, pt_name)
    pt_df = pt_df[["pt_names", "channel_name", "start", "end", "index"]]
    df.drop(columns=["index"], inplace=True)
    pt_df = pt_df.merge(df, on=["pt_names", "channel_name", "start", "end"])
    pos_index = pt_df[(pt_df[col] == 1)]["index"].values
    neg_index = pt_df[pt_df[col] == 0]["index"].values
    pos_data = data[pos_index]
    neg_data = data[neg_index]
    return pos_data, neg_data

def retrive_patients(df, col):
    pt_names = df["pt_names"].unique()
    data_folder = "./data"
    params = [(data_folder, pt_name, col, df[df["pt_names"] == pt_name]) for pt_name in pt_names]
    with Pool(32) as p:
        ttest_result = p.starmap(retrive_one_patient, params)
    pos_data = []
    neg_data = []
    for r in ttest_result:
        pos_data.append(r[0])
        neg_data.append(r[1])
    return np.concatenate(pos_data), np.concatenate(neg_data)

def t_test_all_samples(df, col):
    pos_data, neg_data = retrive_patients(df, col)
    pos_data = pos_data.reshape(pos_data.shape[0], -1)
    neg_data = neg_data.reshape(neg_data.shape[0], -1)
    ttest = stats.ttest_ind(pos_data, neg_data, axis=0, equal_var=True, alternative="greater")[1]
    ttest = ttest.reshape(64, 64)
    return (ttest < 0.05).astype(int),
