from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from skimage import filters
import os
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# import umap
import copy
import pickle

def eval(labels, preds):
    acc = np.mean(preds == labels)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"acc": acc, "recall": recall, "precision": precision, "f1": f1}
class pred_algo:
    """parent class of all prediction algorithms"""
    def __init__(self):
        pass

    def fit(self,train_df, mu_train):
        pass

    def predict(self, test_df, mu_test)->np.ndarray:
        return np.zeros(len(test_df))

    def eval(self, test_df, mu_test, labels)->dict:
        """evaluate the performance of the algorithm"""
        preds = self.predict(test_df, mu_test)
        # print("pred_counts: {}".format(np.unique(preds, return_counts=True)))
        # print("label_counts: {}".format(np.unique(labels, return_counts=True)))
        acc = np.mean(preds == labels)
        recall = recall_score(labels, preds)
        precision = precision_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"acc": acc, "recall": recall, "precision": precision, "f1": f1}
    
    def save(self, fn):
        np.savez(fn, **{key: self.__dict__[key] for key in self.__dict__.keys()})

    def load(self, fn):
        data = np.load(fn)
        for key in data.keys():
            self.__dict__[key] = data[key]
        return self

class threshold():
    def __init__(self,cluster_algo:str = "gmm",seed:int = 42,num_samples:int = 10000, n_clusters:int = 2):
        self.num_samples = num_samples
    
    def sample_uniform_(self,df,mu):
        np.random.seed(42)
        num_samples = min(self.num_samples, len(df))
        # print("df.shape: {}, mu.shape: {}, num_samples: {}".format(df.shape, mu.shape, num_samples))
        indexs = []
        expected_shape = 0
        for pt in df["pt_names"].unique():
            indexs.append(np.random.choice(df[df["pt_names"] == pt].index, min(num_samples, len(df[df["pt_names"] == pt].index)), replace=False))
            # print("indexs[-1].shape", indexs[-1].shape,"len:", df[df['pt_names'] == pt].shape)
            expected_shape += indexs[-1].shape[0]
        indexs = np.concatenate(indexs, 0)
        # print(indexs.shape,"expected_shape: {}".format(expected_shape))
        df_sampled, mu_sampled = df.iloc[indexs].copy(), mu[indexs].copy()
        # print("df_sampled shape: {} mu_sampled shape: {}".format(df_sampled.shape, mu_sampled.shape))
        return df.iloc[indexs], mu[indexs]
    
    def fit(self,train_df, mu_train):

        train_df, _ = self.sample_uniform_(train_df, mu_train)
        reconstructions = train_df["reconstruct"].values
        #use otsu thresholding
        self.threshold = filters.threshold_otsu(reconstructions.reshape(1,-1))
        # print(reconstructions.shape)
        # self.threshold = np.mean(train_df["reconstruct"].values)+np.std(train_df["reconstruct"].values)
        # print("threshold: {}".format(self.threshold))

    def predict(self, test_df, mu_test):
        return test_df["reconstruct"].values < self.threshold
    
    def eval(self, test_df, mu_test, labels)->dict:
        """evaluate the performance of the algorithm"""
        preds = ~self.predict(test_df, mu_test)
        # print("pred_counts: {}".format(np.unique(preds, return_counts=True)))
        # print("label_counts: {}".format(np.unique(labels, return_counts=True)))
        acc = np.mean(preds == labels)
        recall = recall_score(labels, preds)
        precision = precision_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"acc": acc, "recall": recall, "precision": precision, "f1": f1}
    
class artifact_cluster:
    def __init__(self,cluster_algo:str = "gmm",seed:int = 42,num_samples:int = 10000, n_clusters:int = 2):
        case = cluster_algo.lower()
        self.n_clusters = n_clusters
        if case == "kmeans":
            self.cluster_algo = KMeans(n_clusters=self.n_clusters, random_state=seed)
        elif case == "gmm":
            self.cluster_algo = GaussianMixture(n_components=self.n_clusters, random_state=seed,max_iter=2000,tol=1e-6)
        else:
            raise ValueError("cluster_algo must be kmeans or gmm")
        self.seed = seed
        self.num_samples = num_samples
        
    def fit(self, train_df, mu_train):
        df_uniform, mu_uniform = self.sample_uniform_(train_df, mu_train)
        mu_use_uniform = mu_uniform.copy()
        mu_use_uniform = np.concatenate([mu_use_uniform, df_uniform["reconstruct"].values.reshape(-1,1)], axis = 1)
        self.cluster_algo.fit(mu_use_uniform)
        mu_train_use = mu_train.copy()
        mu_train_use = np.concatenate([mu_train_use, train_df["reconstruct"].values.reshape(-1,1)], axis = 1)
        preds = self.cluster_algo.predict(mu_train_use)
        reconstructions = train_df["reconstruct"].values
        means = [reconstructions[preds == i].mean() for i in range(len(np.unique(preds)))]
        # print("means: {}".format(means))
        #artifact class is the one with the highest mean
        self.artifact_class = np.argmax(means)
        self.artifiact_mean_reconstruction = means[self.artifact_class]
        # df_uniform, mu_uniform = self.sample_uniform_(train_df, mu_train)
        # self.eval(df_uniform, mu_uniform, (df_uniform["real"].values > 0.5).astype(int))
    
    def predict(self, test_df, mu_test):
        mu_test_use = mu_test.copy()
        mu_test_use = np.concatenate([mu_test_use, test_df["reconstruct"].values.reshape(-1,1)], axis = 1)
        preds = self.cluster_algo.predict(mu_test_use)
        return (preds != self.artifact_class) #&(test_df["reconstruct"].values < self.artifiact_mean_reconstruction) #true if not artifact

    def eval(self, test_df, mu_test, labels)->dict:
        """evaluate the performance of the algorithm"""
        preds = ~self.predict(test_df, mu_test)
        # print("pred_counts: {}".format(np.unique(preds, return_counts=True)))
        # print("label_counts: {}".format(np.unique(labels, return_counts=True)))
        acc = np.mean(preds == labels)
        recall = recall_score(labels, preds)
        precision = precision_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"acc": acc, "recall": recall, "precision": precision, "f1": f1, "n": len(labels), "n_pred": len(preds)}
    
    def sample_uniform_(self,df,mu):
        np.random.seed(42)
        num_samples = min(self.num_samples, len(df))
        # print("df.shape: {}, mu.shape: {}, num_samples: {}".format(df.shape, mu.shape, num_samples))
        indexs = []
        expected_shape = 0
        for pt in df["pt_names"].unique():
            indexs.append(np.random.choice(df[df["pt_names"] == pt].index, min(num_samples, len(df[df["pt_names"] == pt].index)), replace=False))
            # print("indexs[-1].shape", indexs[-1].shape,"len:", df[df['pt_names'] == pt].shape)
            expected_shape += indexs[-1].shape[0]
        indexs = np.concatenate(indexs, 0)
        # print(indexs.shape,"expected_shape: {}".format(expected_shape))
        df_sampled, mu_sampled = df.iloc[indexs].copy(), mu[indexs].copy()
        # print("df_sampled shape: {} mu_sampled shape: {}".format(df_sampled.shape, mu_sampled.shape))
        return df.iloc[indexs], mu[indexs]
    
    def save(self,save_path):
        #save the cluster_algo
        with open(os.path.join(save_path, "cluster_algo.pkl"), "wb") as f:
            pickle.dump({"cluster_algo": self.cluster_algo, "artifact_class": self.artifact_class}, f)
class pathologic_cluster:
    def __init__(self,cluster_algo:str = "gmm",n_clusters:int = 2,seed:int = 42,num_samples:int = 1000):
        case = cluster_algo.lower()
        self.n_clusters = 2
        if case == "kmeans":
            self.cluster_algo = KMeans(n_clusters=self.n_clusters, random_state=seed)
        elif case == "gmm":
            self.cluster_algo = GaussianMixture(n_components=self.n_clusters, random_state=seed,max_iter=2000,tol=1e-6) 
        else:
            raise ValueError("cluster_algo must be kmeans or gmm")
        self.seed = seed
        self.num_samples = num_samples
        # self.cluster_true_label = 1
    def fit(self,train_df,mu_train):
        mu_train_use = mu_train.copy()
        mu_train_use = np.concatenate([mu_train_use, train_df["reconstruct"].values.reshape(-1,1)], axis = 1)
        _, mu_train_sampled = self.sample_uniform_(train_df, mu_train_use)
        # self.identify_pathologic(train_df, mu_train, patient_wise = True)
        # df_uniform, mu_uniform = self.sample_uniform_(train_df, mu_train)
        # self.eval(df_uniform, mu_uniform, (df_uniform["spike"].values > 0.5).astype(int))
        self.cluster_algo.fit(mu_train_sampled)
        self.identify_pathologic(train_df, mu_train_use, patient_wise = True)
        

    def identify_pathologic(self, train_df, mu_train,patient_wise:bool = False):
        #group by pt_names
        resected_pts = []
        for pt in train_df["pt_names"].unique():
            #if it is resected 
            if np.sum(train_df[train_df["pt_names"] == pt]["remove"].values) > 0:
                #if the number of observations is greater than 50
                if len(train_df[train_df["pt_names"] == pt]) > 50:
                    resected_pts.append(pt)
        #get the preds for the resected patients
        index_resected = train_df["pt_names"].isin(resected_pts)
        preds_resected = self.cluster_algo.predict(mu_train[index_resected])
        resected_df = train_df[index_resected].copy()
        #add the preds
        resected_df["pred"] = preds_resected
        #if patient wise
        if patient_wise:
            ratios = np.zeros(np.unique(preds_resected).shape[0])
            for pt in resected_df["pt_names"].unique():
                #get the preds for the patient
                preds = resected_df[resected_df["pt_names"] == pt]["pred"].values
                preds_resected = resected_df[(resected_df["pt_names"] == pt)&(resected_df["remove"] == 1)]["pred"].values
                #get the number of preds for each cluster
                n_preds = np.zeros(np.unique(preds_resected).shape[0])
                for i in range(np.unique(preds_resected).shape[0]):
                    n_preds[i] = np.sum(preds_resected == i)/np.sum(preds == i) if np.sum(preds == i) > 0 else 0
                #get the ratio
                ratios += n_preds
            ratios /= len(resected_df["pt_names"].unique())
            # #get the cluster with the highest ratio
            # self.pathologic_class = np.argmax(ratios)
        else:
            #get the number of preds for each cluster
            preds_resected = resected_df[resected_df["remove"] == 1]["pred"].values
            preds = resected_df["pred"].values
            ratios = np.array([np.sum(preds_resected == i)/np.sum(preds == i) for i in range(self.n_clusters)])
        
        #get the cluster with the highest ratio
        print("ratios: {}".format(ratios))
        self.pathologic_class = np.argmax(ratios)
            

    def predict(self, test_df, mu_test):
        mu_test_use = mu_test.copy()
        mu_test_use = np.concatenate([mu_test_use, test_df["reconstruct"].values.reshape(-1,1)], axis = 1)
        preds = self.cluster_algo.predict(mu_test_use)
        return preds == self.pathologic_class
    
    def sample_uniform_(self,df,mu):
        np.random.seed(42)
        num_samples = min(self.num_samples, len(df))
        # print("df.shape: {}, mu.shape: {}, num_samples: {}".format(df.shape, mu.shape, num_samples))
        indexs = []
        expected_shape = 0
        for pt in df["pt_names"].unique():
            indexs.append(np.random.choice(df[df["pt_names"] == pt].index, min(num_samples, len(df[df["pt_names"] == pt].index)), replace=False))
            # print("indexs[-1].shape", indexs[-1].shape,"len:", df[df['pt_names'] == pt].shape)
            expected_shape += indexs[-1].shape[0]
        indexs = np.concatenate(indexs, 0)
        # print(indexs.shape,"expected_shape: {}".format(expected_shape))
        df_sampled, mu_sampled = df.iloc[indexs].copy(), mu[indexs].copy()
        # print("df_sampled shape: {} mu_sampled shape: {}".format(df_sampled.shape, mu_sampled.shape))
        return df.iloc[indexs], mu[indexs]
    
    def eval(self, test_df, mu_test, labels)->dict:
        """evaluate the performance of the algorithm"""
        preds = self.predict(test_df, mu_test)
        # print("pred_counts: {}".format(np.unique(preds, return_counts=True)))
        # print("label_counts: {}".format(np.unique(labels, return_counts=True)))
        acc = np.mean(preds == labels)
        recall = recall_score(labels, preds)
        precision = precision_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"acc": acc, "recall": recall, "precision": precision, "f1": f1}
    
    def save(self,path):
        #save the cluster_algo
        with open(os.path.join(path, "cluster_algo.pkl"), "wb") as f:
            pickle.dump({"cluster_algo": self.cluster_algo, "pathologic_class": self.pathologic_class}, f)

class overall_clustering:
    def __init__(self,train_path, artifact_rejection_method:str = "cluster", 
                 artifact_predictor_kwargs:dict = {"cluster_algo": "gmm", "n_clusters": 2, "seed": 0, "num_samples":1000},
                 pathological_predictor_kwargs:dict = {"cluster_algo": "gmm", "n_clusters": 2, "seed": 0, "num_samples": 1000}):
        train_df, mu_train = self.load_(train_path)
        self.train_df = train_df
        self.mu_train = mu_train
        test_path = train_path.replace("train", "test")
        self.test_df, self.mu_test = self.load_(test_path)

        if artifact_rejection_method == "threshold":
            self.artifact_rejector = threshold(**artifact_predictor_kwargs)
        elif artifact_rejection_method == "cluster":
            self.artifact_rejector = artifact_cluster(**artifact_predictor_kwargs)
        else:
            raise ValueError("artifact_rejection_method must be threshold or cluster")
        
        self.pathological_predictor = pathologic_cluster(**pathological_predictor_kwargs)

    def print_eval(self,eval):
        for key in eval.keys():
            print("{}: {}".format(key, round(eval[key],3)), end = " |")
        print("")
    def __call__(self):

        #fit the artifact rejector
        self.artifact_rejector.fit(self.train_df, self.mu_train)
        # #predict the artifact
        artifact_labels_train = (self.train_df["real"].values < 0.5).astype(int)
        artifact_labels_test = (self.test_df["real"].values < 0.5).astype(int)
        eval_vals_train = self.artifact_rejector.eval(self.train_df, self.mu_train, artifact_labels_train)
        print("artifact train: ", end = "")
        self.print_eval(eval_vals_train)
        eval_vals_test = self.artifact_rejector.eval(self.test_df, self.mu_test, artifact_labels_test)
        print("artifact test: ", end = "")
        self.print_eval(eval_vals_test)
        #predict the pathological
        train_df_artifact_rejected = self.train_df[self.artifact_rejector.predict(self.train_df, self.mu_train)]
        mu_train_artifact_rejected = self.mu_train[self.artifact_rejector.predict(self.train_df, self.mu_train)]
        test_df_artifact_rejected = self.test_df[self.artifact_rejector.predict(self.test_df, self.mu_test)]
        mu_test_artifact_rejected = self.mu_test[self.artifact_rejector.predict(self.test_df, self.mu_test)]

        #reindex
        train_df_artifact_rejected.index = np.arange(len(train_df_artifact_rejected))
        test_df_artifact_rejected.index = np.arange(len(test_df_artifact_rejected))
        self.pathological_predictor.fit(train_df_artifact_rejected, mu_train_artifact_rejected)
        eval_vals_train = self.pathological_predictor.eval(train_df_artifact_rejected, mu_train_artifact_rejected, (train_df_artifact_rejected["spike"].values > 0.5).astype(int))
        print("pathological train: ", end = "")
        self.print_eval(eval_vals_train)
        eval_vals_test = self.pathological_predictor.eval(test_df_artifact_rejected, mu_test_artifact_rejected, (test_df_artifact_rejected["spike"].values > 0.5).astype(int))
        print("pathological test: ", end = "")
        self.print_eval(eval_vals_test)

    def predict(self, df, mu,with_artifact:bool = False):
        artifact = self.artifact_rejector.predict(df, mu)
        # artifact = (df["real"] > 0.5).values
        spike = (self.pathological_predictor.predict(df, mu) & artifact)
        if with_artifact:
            # return - (~self.artifact_rejector.predict(df, mu)).astype(int)
            return spike.astype(int) - (~artifact).astype(int)
        return spike & artifact

    def save_preds(self,train_fn, save_suffix = "save"):
        base_dir = os.path.dirname(train_fn)
        save_dir = os.path.join(base_dir, save_suffix)
        os.makedirs(save_dir, exist_ok=True)
        data_train = np.load(train_fn)
        data_train_use = {}
        for keys in data_train.keys():
            if keys not in ["in","out"]:
                data_train_use[keys] = data_train[keys]
        #assert that data_train_use['labels'][:, -2] is the same as self.train_df["real"].values
        assert np.all(data_train_use['labels'][:, -2] == self.train_df["real"].values), "data_train_use['labels'][:, -2] is not the same as self.train_df['real'].values"
        #add the preds
        data_train_use["pred"] = self.predict(self.train_df, self.mu_train)
        data_train_use["pred_with_artifact"] = self.predict(self.train_df, data_train_use["mu"], with_artifact=True)
        data_train_use["pred_prob"] = np.zeros(data_train_use["pred"].shape)
        print("artifact train: ", end = "")
        self.print_eval(eval(data_train_use['labels'][:, -2]<0.5, (data_train_use["pred_with_artifact"]==-1).astype(int)))
        # raise ValueError("stop")
        print("spike train: ", end = "")
        self.print_eval(eval((data_train_use['labels'][:, -2]>0.5)&(data_train_use['labels'][:, -1]>0.5), (data_train_use["pred_with_artifact"]==1).astype(int)))

        np.savez(os.path.join(save_dir, "train_overall_"), **data_train_use)
        data_test = np.load(train_fn.replace("train", "test"))
        data_test_use = {}
        for keys in data_test.keys():
            if keys not in ["in","out"]:
                data_test_use[keys] = data_test[keys]
        #add the preds
        data_test_use["pred"] = self.predict(self.test_df, self.mu_test)
        data_test_use["pred_with_artifact"] = self.predict(self.test_df, data_test_use["mu"], with_artifact=True)
        data_test_use["pred_prob"] = np.zeros(data_test_use["pred"].shape)
        print("artifact test: ", end = "")
        self.print_eval(eval(data_test_use['labels'][:, -2]<0.5, (data_test_use["pred_with_artifact"]==-1).astype(int)))
        print("spike test: ", end = "")
        self.print_eval(eval((data_test_use['labels'][:, -2]>0.5)&(data_test_use['labels'][:, -1]>0.5), (data_test_use["pred_with_artifact"]==1).astype(int)))
        np.savez(os.path.join(save_dir, "test_overall_"), **data_test_use)
        # raise Val/ueError("stop")

    def load_(self,fn):
        data = np.load(fn)
        df = pd.DataFrame({
        "pt_names": data['pt_names'],
        "channel_names": data['channel_names'],
        "starts": data['starts'],
        "ends": data['ends'],
        "detector": data['detector'],
        "spike": data['labels'][:, -1],
        "real": data['labels'][:, -2],
        "remove": data['labels'][:, -3],
        "reconstruct": data['reconstruct']
        })
        mu = data["mu"]
        return df, mu
    
if __name__ == "__main__":
    import sys
    suffix = sys.argv[1] 
    save_suffix = sys.argv[2]
    epoch =  int(sys.argv[3])
    fns = glob.glob(f"res/{suffix}/fold_*/train_{epoch}.npz")
    n_sample_artifact, n_sample_pathologic = save_suffix.split("_")[:2]
    n_sample_pathologic = int(n_sample_pathologic)
    n_sample_artifact = int(n_sample_artifact)
    artifact_predictor_kwargs = {"cluster_algo": "gmm", "n_clusters": 2, "seed": 42, "num_samples":n_sample_artifact}
    pathological_predictor_kwargs = {"cluster_algo": "gmm", "n_clusters": 2, "seed": 42, "num_samples": n_sample_pathologic}
    for fn in sorted(fns):
        algo = overall_clustering(fn, artifact_rejection_method="cluster", artifact_predictor_kwargs=artifact_predictor_kwargs, pathological_predictor_kwargs=pathological_predictor_kwargs)
        algo()
        algo.save_preds(fn, save_suffix = save_suffix)
        print("")