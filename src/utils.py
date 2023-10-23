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

def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model 
        to generalize and improves the interpretability of the model.

    Inputs: 
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output: 
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])
    print(drop_cols)
    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    print(drops)
    x = x.drop(columns=drops)

    return x, drops


