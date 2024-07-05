# PyTorch imports
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Other libraries for data manipulation and visualization
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
from tqdm import tqdm
# import Pool
from multiprocessing import Pool
from itertools import repeat
from sklearn.model_selection import KFold

class HFODataset(Dataset):
    def __init__(self, loaded, params, uniform_sample = False, seed = 42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.patient_name = loaded["patient_name"]       
        feature = loaded["feature"]
        # normalize to 0~1 for each axis 0 (each channel)
        # feature is numpy array (n, 64, 64)
        feature = feature.reshape(feature.shape[0], -1)
        feature = feature - np.min(feature, axis=1, keepdims=True)
        feature = feature / np.max(feature, axis=1, keepdims=True)
        feature = feature.reshape(feature.shape[0], 64, 64)
        feature = torch.tensor(feature).float()
        # check if there is nan
        assert torch.sum(torch.isnan(feature)) == 0

        self.feature = feature
        self.soz = loaded["soz"].astype(int)
        self.removed = loaded["removed"].astype(int)
        self.remove_label = loaded["label"].astype(int)
        self.hfo_type = loaded["hfo_type"]
        self.artifact = loaded["artifact"]
        self.spike = loaded["spike"]
        self.labels = np.concatenate([self.remove_label[:,None], self.soz[:,None], self.removed[:,None], self.artifact[:,None], self.spike[:,None]], -1).astype(np.float32)
        self.channel_name = loaded["channel"]
        self.start = loaded["start"]
        self.end = loaded["end"]
        self.dataset_size = params["dataset_size"]
        self.uniform_sample = uniform_sample
        self.time_augmentation = params["time_augmentation"]
        self.weight = 2

    def __len__(self):
        if self.uniform_sample:
            return min(self.dataset_size, self.feature.shape[0])
        else:
            return self.feature.shape[0]

    def __getitem__(self, ind):
        if self.uniform_sample and (self.dataset_size < len(self.feature)):
            random_ind = random.randint(0, len(self.feature)-1)
        else:
            random_ind = ind
        #print("random_ind : ", random_ind, "len : ", len(self.feature))
        feature = self.feature[random_ind]
        label = self.labels[random_ind]
        channel_name = self.channel_name[random_ind]
        start = self.start[random_ind]
        end = self.end[random_ind]
        if self.time_augmentation:
            if random.random() < 0.5:
                feature = torch.flip(feature, dims=[-1])
        identifier = self.patient_name + "_" + channel_name + "_" + str(start) + "_" + str(end)
        return self.patient_name, feature, torch.tensor(label), channel_name, start, end, self.hfo_type[random_ind], identifier

def retrive_data(patient_folder):
    res = {}
    patient_name = patient_folder.split("/")[-1]
    label = pd.read_csv(patient_folder + "/label.csv")
    if "SOZ" not in label.columns:
        label["SOZ"] = 0
    if "Resection" not in label.columns:
        label["Resection"] = 0
    if "removed" not in label.columns:
        label["removed"] = 0
    if "label" not in label.columns:
        label["label"] = 0
    res["soz"] = label["SOZ"].values
    res["removed"] = label["Resection"].values
    res["label"] = label["label"].values
    res["patient_name"] = patient_name
    res["channel"] = label["channel_name"].values
    res["start"] = label["start"].values
    res["end"] = label["end"].values
    res["spike"] = label["spike"].values
    res["artifact"] = label["artifact"].values
    res["duration"] = label["duration"].values
    data_load = np.load(patient_folder + "/feature.npz", allow_pickle=True)
    res["feature"] = data_load["feature"].astype(np.float32)
    res["hfo_type"] = np.array(label["detector_type"])


    # only keep the hfo that artifact > 0
    index = np.where((res["artifact"] >= 0) & (res["duration"] < 150))[0]
    res["feature"] = res["feature"][index]
    res["soz"] = res["soz"][index]
    res["removed"] = res["removed"][index]
    res["label"] = res["label"][index]
    res["channel"] = res["channel"][index]
    res["start"] = res["start"][index]
    res["end"] = res["end"][index]
    res["spike"] = res["spike"][index]
    res["artifact"] = res["artifact"][index]
    res["hfo_type"] = res["hfo_type"][index]
    res["duration"] = res["duration"][index]
    
    return res



def create_kfold_index(patient_list, kth, seed, K=5):
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    for i, (train_index, test_index) in enumerate(kf.split(patient_list)):
        if i == kth:
            return test_index
        
def create_kfold_patient_and_dataset(patient_list, dataset_list, kth, seed, K=5):
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    for i, (train_index, test_index) in enumerate(kf.split(patient_list)):
        if i == kth:
            return patient_list[train_index],dataset_list[train_index],patient_list[test_index], dataset_list[test_index]

def retrive_one_patient(selected_patient, dataset_name, data_dir):
    if dataset_name == "detroit":
        data_dir = os.path.join(data_dir, "detroit")
    elif dataset_name == "ucla":
        data_dir = os.path.join(data_dir, "ucla")
    elif dataset_name == "seeg":
        data_dir = os.path.join(data_dir, "seeg")
    patient_folder = os.path.join(data_dir, selected_patient)
    loaded = retrive_data(patient_folder)
    return loaded

def create_one_patient_dataset(selected_patient, dataset_name, data_dir, dataset_params, uniform_sample,seed = 42):
    loaded = retrive_one_patient(selected_patient, dataset_name, data_dir)
    dataset = HFODataset(loaded, dataset_params, uniform_sample=uniform_sample, seed=seed)
    return dataset 

def create_dataset_multi_process(patient_list, dataset_names, data_dir , dataset_params, uniform_sample ,n_jobs = 1):
    dataset_list, weight_list = [], []
    for patient, dataset in zip(patient_list, dataset_names):
        dataset = create_one_patient_dataset(patient, dataset, data_dir, dataset_params, uniform_sample)
        dataset_list.append(dataset)
        weight_list.append(dataset.weight)
    return dataset_list, np.nanmedian(weight_list)

def select_patients(meta_fn, resection=True):
    meta = pd.read_csv(meta_fn)
    meta = meta.sort_values(by="pt_name")
    if resection:
        meta = meta[meta["resection-status"] == 1]
    selected_patients =meta["pt_name"].values
    dataset_name = meta["dataset"].values
    return selected_patients, dataset_name

def split_patients(meta_fn):
    #split into ucla resected, detrioit resected, and non resected
    meta = pd.read_csv(meta_fn)
    ucla_resected = meta[(meta["resection-status"]==1) & (meta["dataset"]=="ucla")]["pt_name"].values
    ucla_resected_dataset = meta[(meta["resection-status"]==1) & (meta["dataset"]=="ucla")]["dataset"].values

    detroit_resected = meta[(meta["resection-status"]==1) & (meta["dataset"]=="detroit")]["pt_name"].values
    detroit_resected_dataset = meta[(meta["resection-status"]==1) & (meta["dataset"]=="detroit")]["dataset"].values

    non_resected = meta[(meta["resection-status"]==0)]["pt_name"].values
    non_resected_dataset = meta[(meta["resection-status"]==0)]["dataset"].values

    return ucla_resected, ucla_resected_dataset, detroit_resected, detroit_resected_dataset, non_resected, non_resected_dataset

def create_loader_multi_processing(k, args, train= True, train_uniform_sample=True):

    ucla_resected, ucla_resected_dataset, detroit_resected, detroit_resected_dataset, non_resected, non_resected_dataset = split_patients(args["meta_fn"])

    train_pts = []
    test_pts = []

    train_dataset_names = []
    test_dataset_names = []

    for pt_names,datasets  in zip([ucla_resected, detroit_resected, non_resected], [ucla_resected_dataset, detroit_resected_dataset, non_resected_dataset]):
        train_pt_names,train_datasets, test_pt_names, test_datasets = create_kfold_patient_and_dataset(pt_names, datasets, k, args["seed"], K=args["K"])
        train_pts.append(train_pt_names)
        test_pts.append(test_pt_names)
        train_dataset_names.append(train_datasets)
        test_dataset_names.append(test_datasets)
    
    train_pts = np.concatenate(train_pts)
    test_pts = np.concatenate(test_pts)

    train_dataset_names = np.concatenate(train_dataset_names)   
    test_dataset_names = np.concatenate(test_dataset_names)

    train_indexs = np.arange(len(train_pts))
    np.random.seed(k)
    np.random.shuffle(train_indexs)

    val_index = train_indexs[:30]
    train_index = train_indexs[30:]

    val_pt_names = train_pts[val_index]
    train_pt_names = train_pts[train_index]

    val_dataset_names = train_dataset_names[val_index]
    train_dataset_names = train_dataset_names[train_index]
    if train:
        ## Train
        #print(all_pt_names[train_index], all_pt_names[val_index])
        train_dataset, _ = create_dataset_multi_process(train_pt_names, train_dataset_names, args["data_dir"], args, uniform_sample = train_uniform_sample)   
        train_dataset = data.ConcatDataset(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=1, pin_memory=True)
        val_dataset, _= create_dataset_multi_process(val_pt_names, val_dataset_names, args["data_dir"], args, uniform_sample = train_uniform_sample)
        val_dataset = data.ConcatDataset(val_dataset)
        val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=True, num_workers=1, pin_memory=True)
        print("loader created")
        return train_loader, val_loader
    else:
        ## test
        #print(all_pt_names[test_index])
        args["time_augmentation"] = False
        test_dataset, _ = create_dataset_multi_process(test_pts, test_dataset_names, args["data_dir"], args, uniform_sample = False)
        test_dataset = data.ConcatDataset(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)
        return test_loader

if __name__ == "__main__":
    import sys
    import os
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)
    import src.param as param
    args = param.get_args()
    train_loader,val_loader = create_loader_multi_processing(2, args, train= True, train_uniform_sample=True)
    pt_names = []   
    channel_names = []
    starts = []
    ends = []
    detectors = []
    for batch in  train_loader:
        pt_name, _,_, channel_name, start, end, detector,_ = batch
        pt_names  += pt_name
        channel_names += channel_name   
        starts += start
        ends += end 
        detectors += detector
    
    for batch in  val_loader:
        pt_name, _,_, channel_name, start, end, detector,_ = batch
        pt_names  += pt_name
        channel_names += channel_name   
        starts += start
        ends += end 
        detectors += detector
    
    HFOs = pd.DataFrame({"pt_name":pt_names, "channel_name":channel_names, "start":starts, "end":ends, "detector":detectors})   
    print(HFOs[HFOs.duplicated()])

