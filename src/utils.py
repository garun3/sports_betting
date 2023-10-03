import numpy as np
from cache_pandas import cache_to_csv
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score
import xgboost as xgb
import pickle
from datetime import date as d, timedelta
from understatapi import UnderstatClient
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def split_data(new_df, drop_cols, pred_cols, target):
    date = str(d.today())
    new_df = new_df.drop(columns=drop_cols, axis=1)
    train = new_df[new_df['Date'] < '2018-08']
    #test = new_df[new_df['Date'] >= '2022-08' and new_df['Date'] < '2022-08']
    test = new_df.loc[(new_df['Date'] >= '2018-08') & (new_df['Date'] < date)]

    pred = new_df.loc[(new_df['Date'] >= date)]

    train = train.select_dtypes(include=np.number)
    test = test.select_dtypes(include=np.number)
    pred1 = pred.select_dtypes(include=np.number)
    
    X_train = train.loc[:, pred_cols]#predictors+new_cols]
    y_train = train.loc[:, target]
    
    X_test = test.loc[:, pred_cols]
    y_test = test.loc[:, target]

    X_pred = pred1.loc[:, pred_cols]
    X_test_pred = X_test
    return X_train, y_train, X_test, y_test, X_pred, pred

