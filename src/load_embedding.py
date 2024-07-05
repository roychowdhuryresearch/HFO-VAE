import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
import glob
import seaborn as sns
from openTSNE import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score
from multiprocessing import Pool
from fit_embedding import sample, load_df

def load_anatomical_locations(path):
    dfs = []
    anatomical_map = pd.read_excel("./data/anatomical/HFO_anatomylabel.xlsx", sheet_name="subcategory")
    abreviation_map =  {"F":"frontal","T":"temporal","P":"parietal","O":"occipital","LIM":"limbic"}
    anatomical_map["subcategory"] = anatomical_map["subcategory"].apply(lambda x: abreviation_map[x] if x in abreviation_map else "NaN")
    #conver the anatomical map to a dictionary
    anatomical_map = dict(zip(anatomical_map["label"],anatomical_map["subcategory"]))
    for fn in glob.glob(os.path.join(path,"*/*.xlsx")):
        # print(fn)
        df = pd.read_excel(fn)
        #get the dataset
        dataset = fn.split("/")[-2]
        #get the patient name
        pt_name = fn.split("/")[-1].split(".")[0].split("_electrode_profile")[0]
        #add the dataset and patient name to the dataframe
        df["dataset"] = dataset
        df["pt_names"] = pt_name
        #remove the anatomy_2 and Override columns
        if "anatomy_2" in df.columns:
            df = df.drop(columns=["anatomy_2","Override"])
        #rename ch to channel_names
        df = df.rename(columns={"ch":"channel_name"})
        #map the anatomical locations
        if "anatomy" not in df.columns:
            df["anatomy"] = None
        df["anatomy"] = df["anatomy"].apply(lambda x: anatomical_map[x] if x in anatomical_map else "NaN")
        dfs.append(df)
    df = pd.concat(dfs) 
    return df

def load_pathology(path):
    df = pd.read_csv(path)
    #save only the columns we need
    df = df[["number","Yipeng_proj"]]
    #rename the columns
    df.columns = ["pt_names","pathology"]
    #replace dysplasia with FCD
    df["pathology"] = df["pathology"].replace({"dysplasia":"FCD","Dysplasia":"FCD"})
    return df

def load_meta(path):
    meta_df = pd.read_csv(path)
    #rename pt_name to pt_names
    meta_df = meta_df.rename(columns={"pt_name":"pt_names"})
    return meta_df

def merge_dataframes(df_embedding,df_anatomical, df_pathology,df_labels,df_meta):
    df_use = df_embedding[["pt_names","channel_name","start","end", "detector","mu_embed_0","mu_embed_1"]]
    hfos = df_use[["pt_names","channel_name","start","end","detector"]]
    #make index a column
    df_use = df_use.reset_index()
    #same for df_labels
    df_labels = df_labels.reset_index() 
    previous_shape = df_use.shape
    hfos = df_labels[["pt_names","channel_name","start","end","detector"]]
    df_use = df_use.merge(df_labels, on=["index","pt_names","channel_name","start","end", "detector"], how="inner")
    hfos = df_use[["pt_names","channel_name","start","end","detector"]]
    assert df_use.shape[0] == previous_shape[0], "df_use.shape: {}, previous_shape: {}".format(df_use.shape, previous_shape)
    df_use = df_use.merge(df_anatomical, on=["pt_names","channel_name"], how="left")
    assert df_use.shape[0] == previous_shape[0], "df_use.shape: {}, previous_shape: {}".format(df_use.shape, previous_shape)
    #print("df_use.shape after merge with anatomical:",df_use.shape)
    #replace the NaN values with "NaN"
    df_use["anatomy"] = df_use["anatomy"].fillna("NaN")
    #merge the pathology dataframe
    df_use = df_use.merge(df_pathology, on="pt_names", how="left")
    assert df_use.shape[0] == previous_shape[0], "df_use.shape: {}, previous_shape: {}".format(df_use.shape, previous_shape)
    df_use["pathology"] = df_use["pathology"].fillna("NaN")
    df_meta.drop(columns=["dataset"], inplace=True)
    df_use = df_use.merge(df_meta, on="pt_names", how="left")
    assert df_use.shape[0] == previous_shape[0], "df_use.shape: {}, previous_shape: {}".format(df_use.shape, previous_shape)
    #merge the labels dataframe
    return df_use

def create_df(path:str,label_file:str, embedding_file:str):
    meta = load_meta("./data/meta.csv")
    pathology = load_pathology("./data/Pathology_HFOstudy.csv")
    anatomical = load_anatomical_locations("./data/anatomical")
    embedding = pd.read_csv(os.path.join(path,embedding_file))
    labels,_ = load_df(os.path.join(path,label_file))
    df = merge_dataframes(embedding,anatomical, pathology,labels,meta)
    #print("df.shape",df.shape,"embedding.shape",embedding.shape,"anatomical.shape",anatomical.shape,"pathology.shape",pathology.shape,"labels.shape",labels.shape,"meta.shape",meta.shape)
    df["vae-predict"] = df["pred"].replace({-1:"Artifact", 0: "Physiological", 1: "Pathological"})
    # for col spk-HFO convert 1 as HFO-with-spike and 0 as HFO-without-spike
    df["real_label"] = (df["artifact"] > 0.5)
    df["spike_label"] = (df["spike"] > 0.5) & (df["artifact"] > 0.5)
    # real label = 0 is 0, spike label = 1 and real label = 1 is 2, spike label = 0 is and real label = 1 is 1
    df["labels"] = df["real_label"].astype(int) + df["spike_label"].astype(int)
    df["spk-HFO"] = df["labels"].replace({2: "HFO-with-spike", 0: "Artifact", 1: "HFO-without-spike"})
    # random_idx = np.random.choice(len(df_p), min(random_sample_df_p_n, len(df_p)), replace=False)
    df["Projected Dimension 1"] = df["mu_embed_0"]
    df["Projected Dimension 2"] = df["mu_embed_1"]
    return df


if __name__ == "__main__":
    #load the dataframes
    path = "./res/2023-11-02_21/fold_2/10000_1000_51"
    embedding = "train_embedding.csv"
    label_path = "train_overall_.npz"
    df_train = create_df(path,label_path,embedding)
    df_test = create_df(path,label_path.replace("train","test"),embedding.replace("train","test"))
    
    df_train_hfos = df_train[['pt_names', 'channel_name', 'start', 'end']]
    print("len(df_train_hfos)",len(df_train_hfos))
    df_test_hfos = df_test[['pt_names', 'channel_name', 'start', 'end']]
    print("len(df_test_hfos)",len(df_test_hfos))
    #first assert that the train and test set rows are unique
    # assert len(df_train_hfos) == len(df_train_hfos.drop_duplicates()), "train set has duplicate rows, duplicate hfos: {}".format(df_train_hfos[df_train_hfos.duplicated()])
    # assert len(df_test_hfos) == len(df_test_hfos.drop_duplicates()), "test set has duplicate rows, duplicate hfos: {}".format(df_test_hfos[df_test_hfos.duplicated()])
    #now assert that the train and test sets are unique
    assert len(df_train_hfos.merge(df_test_hfos, how="inner")) == 0, "train and test sets are not unique, common hfos: {}".format(df_train_hfos.merge(df_test_hfos, how="inner"))




