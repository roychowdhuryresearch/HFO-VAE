import pandas as pd

# logstic regression
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# decision tree 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, f1_score
date = "2023-12-28_2"
suffix = "10000_2000_81"
import os

# date = "2023-11-02_21"
# suffix = "10000_2000_81"
for th in np.arange(0,60,5):
#for th in [0]:
    fn = f'./res/{date}/{suffix}/auc_kfold_{th}.xlsx'
    # return a dict of dataframes
    df = pd.read_excel(fn, engine='openpyxl', sheet_name=None)
    col_list = [["r_hfo"], ["r_real"],["r_spike"],["r_pred"],
                ["Age_yr", "Male","r_hfo"], ["Age_yr", "Male", "r_real"],["Age_yr", "Male", "r_spike", "n_spike"],["Age_yr", "Male", "r_pred", "n_pred"],
                ["soz_resected"], 
                ["Age_yr", "Male"],
                ["Age_yr", "Male", "r_pred"],
                ["Age_yr", "Male", "r_spike"],
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
    df_res = []
    for col in col_list:
        acc_list, f1_list = [], []
        acc_detroit, f1_detroit = [], []
        acc_ucla, f1_ucla = [], []
        for i in range(5):
            df_train = df[f"train_{i}"]
            df_test = df[f"test_{i}"]
            X_train = df_train[col].values
            y_train = df_train["seizure-free"].values
            X_test = df_test[col].values
            y_test = df_test["seizure-free"].values
            # clf = LogisticRegression(random_state=0, class_weight='balanced', fit_intercept=False, penalty=None)
            #clf = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=4, criterion='gini', n_jobs=40)
            clf = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=None, criterion='gini', class_weight='balanced') 
            #clf = RandomForestClassifier(random_state=42+i, n_estimators=150, class_weight='balanced', max_depth=7) # 100. 5 OK # 150 7 OK
            
            # clf = DecisionTreeClassifier(random_state=42, n_estimators=100, class_weight='balanced', max_depth=None, max_features='sqrt', criterion='gini')
            # clf = SVC(random_state=0, class_weight='balanced', probability=True, C= 20, gamma= 0.01, kernel= 'rbf')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:,1]
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            acc_list.append(acc)
            f1_list.append(f1)
            df_detoit = df_test[df_test["dataset"] == "detroit"]
            pred_detroit = clf.predict(df_detoit[col].values)
            acc_detroit.append(accuracy_score(df_detoit["seizure-free"].values, pred_detroit))
            f1_detroit.append(f1_score(df_detoit["seizure-free"].values, pred_detroit))
            df_ucla = df_test[df_test["dataset"] != "detroit"]
            pred_ucla = clf.predict(df_ucla[col].values)
            acc_ucla.append(accuracy_score(df_ucla["seizure-free"].values, pred_ucla))
            f1_ucla.append(f1_score(df_ucla["seizure-free"].values, pred_ucla))
            if "_".join(col) == "Age_yr_Male_r_pred":
                print(f"{col}, acc: {acc:.3f}, f1: {f1:.3f}")
        #df_res.append(["_".join(col),np.mean(acc_list), np.std(acc_list), np.mean(f1_list), np.std(f1_list), np.mean(acc_detroit), np.std(acc_detroit), np.mean(f1_detroit), np.std(f1_detroit), np.mean(acc_ucla), np.std(acc_ucla), np.mean(f1_ucla), np.std(f1_ucla)])
        df_res.append(["_".join(col),np.mean(acc_list), np.std(acc_list)/5, np.mean(f1_list), np.std(f1_list)/5, np.mean(acc_detroit), np.std(acc_detroit)/5, np.mean(f1_detroit), np.std(f1_detroit)/5, np.mean(acc_ucla), np.std(acc_ucla)/5, np.mean(f1_ucla), np.std(f1_ucla)/5])
    df_final= pd.DataFrame(df_res, columns=["feature","acc_mean", "acc_std", "f1_mean", "f1_std", "acc_detroit_mean", "acc_detroit_std", "f1_detroit_mean", "f1_detroit_std", "acc_ucla_mean", "acc_ucla_std", "f1_ucla_mean", "f1_ucla_std"])
    df_res = df_final.copy()
    df_res["Acc"] = df_res.apply(lambda x: f"{x['acc_mean']:.3f}({x['acc_std']:.3f})", axis=1)
    df_res["F1"] = df_res.apply(lambda x: f"{x['f1_mean']:.3f}({x['f1_std']:.3f})", axis=1)
    df_res["Acc Detroit"] = df_res.apply(lambda x: f"{x['acc_detroit_mean']:.3f}({x['acc_detroit_std']:.3f})", axis=1)
    df_res["F1 Detroit"] = df_res.apply(lambda x: f"{x['f1_detroit_mean']:.3f}({x['f1_detroit_std']:.3f})", axis=1)
    df_res["Acc UCLA"] = df_res.apply(lambda x: f"{x['acc_ucla_mean']:.3f}({x['acc_ucla_std']:.3f})", axis=1)
    df_res["F1 UCLA"] = df_res.apply(lambda x: f"{x['f1_ucla_mean']:.3f}({x['f1_ucla_std']:.3f})", axis=1)
    df_res["feature"] = df_res["feature"].str.replace("r_hfo", "% HFO")
    df_res["feature"] = df_res["feature"].str.replace("r_real", "% Real")
    df_res["feature"] = df_res["feature"].str.replace("r_spike", "% spk-HFO")
    df_res["feature"] = df_res["feature"].str.replace("r_pred", "% Path. HFO")
    df_res["feature"] = df_res["feature"].str.replace("soz_resected", "SOZ Resc.")
    df_res["feature"] = df_res["feature"].str.replace("Age_yr", "Age")
    df_res["feature"] = df_res["feature"].str.replace("Male", "Gender")
    df_res["feature"] = df_res["feature"].str.replace("_", " + ")
    df_ratio = df_res.iloc[[0,1,2,3,9,12,11,10,-2,-1]]
    save_fn = f"./res/{date}/{suffix}/subject_ablation/SEM_RF_predict_{date}_{suffix}_{th}.csv"
    if not os.path.exists(os.path.dirname(save_fn)):
        os.makedirs(os.path.dirname(save_fn)) 
    df_ratio.to_csv(save_fn, index=False)
