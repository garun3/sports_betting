import soccerdata as sd 
import pandas as pd
import os
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

import config
from process import prep_data
from utils import split_data, remove_collinear_features



def random_forest_win(data, predictors, X_train, X_test, y_train, y_test, call_type, target_col, model_type):  
    if model_type == 'RF':
        model = RandomForestClassifier(random_state=1)#n_estimators=50, min_samples_split=7, random_state=1) 
    else: 
        params = {'colsample_bytree': 0.5, 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 5, 'reg_lambda': 0, 'subsample': 0.8}
        model = xgb.XGBClassifier(random_state=1)#**params)#, objective='binary:logistic')

    #model = ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.8, min_samples_leaf=17, min_samples_split=5, n_estimators=100)
    params = {'colsample_bytree': 0.5, 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 5, 'reg_lambda': 0, 'subsample': 0.8}
    #params = {'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.01, 'max_depth': 3, 'reg_lambda': 10, 'subsample': 0.8}

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    #return preds,1,1
    #if call_type == 'predict':
    #    return preds, model
    #combined = pd.DataFrame(dict(actual=test['Target'], prediction=preds), index = test.index)
    #print(test.groupby('GameID').filter(lambda x: x['prediction'].sum()==1))
    precision = precision_score(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    combined = pd.DataFrame(dict(actual=y_test, prediction=preds), index = y_test.index)
    c1 = combined.merge(data[['Date', 'Team', 'Opponent', 'TeamGoals', 'OpponentGoals','GameID', 'Venue']], left_index=True, right_index=True)

    # predictions where both home and away team prediction match
    predictions = c1.groupby('GameID').filter(lambda x: x['prediction'].sum()==1)
    #print(predictions)
    table = c1[['GameID','Date','Team', 'Opponent', 'prediction', 'actual']].merge(c1[['GameID','Date','Team', 'Opponent', 'prediction']], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])
    #table.to_csv(f'{config.PATH}/../Testing/{config.LEAGUE}/{config.LEAGUE}_{target_col}.csv')

    table = table[table['prediction_x'] == table['prediction_y']]#.filter(lambda x: x['prediction_x'] != x['prediction_y'])
    precision1 = precision_score(table['actual'], table['prediction_x'])
    accuracy1 = accuracy_score(table['actual'], table['prediction_x'])

    print(target_col)
    print(precision, accuracy, precision1, accuracy1) 
    preds = model.predict_proba(X_test)
    #preds = list(map(tuple, preds))
    #test = new_df.loc[(new_df['Date'] >= '2018-08') & (new_df['Date'] < date)]


    #test['preds'] = preds
    #table = test.sort_index()[['GameID','Date','Team', 'Opponent', 'preds']].merge(test[['GameID','Date','Team', 'Opponent', 'preds']], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])

    '''
    values_to_write = [f'League-{self.league}',
        f'Target-{target_col}', f'Precision1-{precision1}', 
        f'Accuracy1- {accuracy1}', f'Precision-{precision}', 
        f'Accuracy-{accuracy}']
    with open('models/model_stats.txt', 'w') as f:
        for line in values_to_write:
            f.write(line)
            f.write('\n')
    '''
    
    #precision, accuracy = 1,1
    #precision = cross_val_score(model, train[predictors], train['TotalGoals'], scoring='neg_mean_absolute_error')
    return table, precision1, accuracy1, model, combined, preds

def train(date, pred_type, predictors):
    #new['WinTarget'] = codes.cat.codes
    #cats = codes.cat.categories
    #print(new['WinTarget'], new['Result'])
    #return new, 'hi'
    new = pd.read_csv(f'{config.PATH}/{config.CACHE_PATH}/new.csv')
    new_df = new
    #predictors = ['VenueCode', 'OpponentCode', 'ELODif', 'Day'] + dif_cols
    #predictors = ['VenueCode', 'ELODif']
    #print(new)
    #if pred_type == 'WinTarget':
    #    new_df = new.dropna(subset=['TeamWinOdds'])
    #else:
    new_df = new.dropna(subset=['O2.5', 'TeamWinOdds'] + list(new.filter(like='FIFA').columns))
    #new_df = new.dropna()
    print(new_df.shape, new_df.dropna().shape)

    #new_df = new_df.dropna()
    #print(new_df.shape)
    
    #new_df['TotalGoals'] = new_df['TotalGoals'].astype('int')
    #new['TotalGoals'] = new['TotalGoals'].astype('int')
    X_train, y_train, X_test, y_test, X_pred, pred = split_data(new_df, [], predictors, pred_type)
    # drop collinear features
    #X_train, drop_cols = remove_collinear_features(X_train, 0.9)
    #X_test = X_test.drop(columns=drop_cols)
    #y_train = y_train.drop(columns=drop_cols)
    #y_test = y_test.drop(columns=drop_cols)
    #X_pred = X_pred.drop(columns=drop_cols)

    #s = set(drop_cols)
    #predictors = [x for x in predictors if x not in s]
    #predictors = predictors - drop_cols


    scaling=StandardScaler()
    X_train_scaled=scaling.fit_transform(X_train)
    X_test_scaled=scaling.transform(X_test)
    X_pred_scaled = scaling.transform(X_pred)
    y_train = np.array(y_train)

    pca = PCA(0.85, random_state=1)
    pca.fit(X_train_scaled)
    X_train_scaled_pca = pca.transform(X_train_scaled)
    X_test_scaled_pca = pca.transform(X_test_scaled)
    X_pred_scaled_pca = pca.transform(X_pred_scaled)

    #X_train = X_train_scaled
    #X_test = X_test_scaled
    #X_pred = X_pred_scaled
    #print(X_pred_scaled_pca)

    #train = new_df[new_df['Date'] < '2018-09-18']
    #return train
    #test = new_df[new_df['Date'] >= '2022-08' and new_df['Date'] < '2022-08']
    #new_df.Date
    #test = new_df[(new_df['Date'] >= '2018-09-18') & (new_df['Date'] < '2023-09')]
    #pred = new.loc[(new['Date'] >= date)]
    #pred_type = 'WinTarget'
    table, prec, acc, model, combined, preds = random_forest_win(df_rolling, predictors, X_train, X_test, y_train, y_test, 'predict', pred_type, 'RF')

    test = new_df.loc[(new_df['Date'] >= '2018-08') & (new_df['Date'] < date)]
    '''
    test['preds'] = preds
    test['actual'] = y_test
    table = test.sort_index()[['GameID','Date','Team', 'Opponent', 'preds', 'actual']].merge(test[['GameID','Date','Team', 'Opponent', 'preds', 'actual']], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])
    table.to_csv(f'{config.PATH}/../Testing/{config.LEAGUE}/{config.LEAGUE}_{pred_type}.csv')
    '''
    cols = []
    for i in range(preds.shape[1]):
        new_col = f'class_{i}'
        test[f'class_{i}'] = preds[:,i]
        cols.append(new_col)
    #test['under_prod'] = preds[0]
    #test['over_prod'] = preds[0]
    test['actual'] = y_test
    table = test.sort_index()[['GameID','Date','Team', 'Opponent', 'actual'] + cols].merge(test[['GameID','Date','Team', 'Opponent', 'actual'] + cols], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])
    table.to_csv(f'{config.PATH}/../Testing/{config.LEAGUE}/{config.LEAGUE}_{pred_type}.csv')


    preds1 = model.predict(X_pred)  
    preds = model.predict_proba(X_pred)  
    #preds = list(map(tuple, preds))
    cols = []
    for i in range(preds.shape[1]):
        new_col = f'class_{i}'
        pred[f'class_{i}'] = preds[:,i]
        cols.append(new_col)
    #pred['preds'] = preds
    table = pred.sort_index()[['GameID','Date','Team', 'Opponent'] + cols].merge(pred[['GameID','Date','Team', 'Opponent']+ cols], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])
    table.to_csv(f'{config.PATH}/../Predictions/{config.LEAGUE}/{config.LEAGUE}_{pred_type}.csv')
    filename = f'{config.PATH}/../models/{config.LEAGUE}/{pred_type}_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    #print(combined)
    #preds = model.predict(pred[predictors])
    #c1 = preds.merge(test[['Date', 'Team', 'Opponent', 'TotalGoals', 'GameID']], left_index=True, right_index=True)   
    return table


def predict(predictors, pred_type):
    new = pd.read_csv(f'{config.PATH}/{config.CACHE_PATH}/new.csv')
    filename = f'{config.PATH}/../models/{config.LEAGUE}/{pred_type}_model.sav'
    model = pickle.load(open(filename, 'rb'))
    
    if pred_type == 'WinTarget':
        new_df = new.dropna(subset=['TeamWinOdds'])
    else:
        new_df = new.dropna(subset=['O2.5'])
    
    
    #pred = pred[predictors]
    #print(dict(pred.isna().sum()), pred.shape, pred.dropna().shape, predictors)
    pred = new.loc[(new_df['Date'] >= date)]
    X_train, y_train, X_test, y_test, X_pred, pred = split_data(new_df, [], predictors, pred_type)

    scaling=StandardScaler()
    X_pred_scaled=scaling.fit_transform(X_pred)
    #X_test_scaled=scaling.transform(X_test)
    #y_train = np.array(y_train)

    pca = PCA(0.95)
    pca.fit(X_train_scaled)
    #X_train_scaled_pca = pca.transform(X_train_scaled)
    X_test_scaled_pca = pca.transform(X_test_scaled)

    X_train = X_train_scaled_pca
    X_test = X_test_scaled_pca
    

    preds = model.predict(X_test)
    return preds, pred

if __name__ == "__main__":
    date = str(d.today())#'2023-09-26'
    df_rolling = pd.read_csv(f'{config.PATH}/{config.CACHE_PATH}/rolling.csv')
    #dif_cols = #prep_data(df_rolling, date)#['30/09/23', '01/10/23', '02/10/23'])
    #predictors = ['VenueCode', 'OpponentCode', 'ELODif', 'Day'] + dif_cols
    if config.XG:
        predictors = ['TotalShotsRolling', 'TeamGoalsRolling', 'TeamFoulsRolling', 'GoalsRolling_Dif', 'ShotsonTargetRolling_Dif', 'OpponentShotsonTargetRolling', 'TotalELORolling', 'xGRolling_Dif', 'ShotsRolling_Dif', 'YellowCardsRolling_Dif', 'TeamRedCardsRolling', 'TotalRedCardsRolling', 'OpponentYellowCardsRolling', 'ELODif', 'TotalFoulsRolling', 'TeamShotsonTargetRolling', 'OpponentCornersRolling', 'TotalCornersRolling', 'FoulsRolling_Dif', 'OpponentGoalsRolling', 'RedCardsRolling_Dif', 'TotalGoalsRolling', 'TeamShotsRolling', 'CornersRolling_Dif', 'TeamCornersRolling', 'OpponentCode', 'TotalYellowCardsRolling', 'VenueCode', 'OpponentRedCardsRolling', 'OpponentShotsRolling', 'TeamYellowCardsRolling', 'TotalShotsonTargetRolling', 'DayCode', 'OpponentFoulsRolling']
        preds_ = predictors + ['TotalxGRolling', 'TeamxGRolling', 'OpponentxGRolling'] 
    else:
        predictors = ['TotalShotsRolling', 'TeamGoalsRolling', 'TeamFoulsRolling', 'GoalsRolling_Dif', 'ShotsonTargetRolling_Dif', 'OpponentShotsonTargetRolling', 'TotalELORolling', 'ShotsRolling_Dif', 'YellowCardsRolling_Dif', 'TeamRedCardsRolling', 'TotalRedCardsRolling', 'OpponentYellowCardsRolling', 'ELODif', 'TotalFoulsRolling', 'TeamShotsonTargetRolling', 'OpponentCornersRolling', 'TotalCornersRolling', 'FoulsRolling_Dif', 'OpponentGoalsRolling', 'RedCardsRolling_Dif', 'TotalGoalsRolling', 'TeamShotsRolling', 'CornersRolling_Dif', 'TeamCornersRolling', 'OpponentCode', 'TotalYellowCardsRolling', 'VenueCode', 'OpponentRedCardsRolling', 'OpponentShotsRolling', 'TeamYellowCardsRolling', 'TotalShotsonTargetRolling', 'DayCode', 'OpponentFoulsRolling']

    #predictors = ['VenueCode', 'OpponentCode', 'DayCode'] 
    preds_ = predictors + ['TeamELO', 'OpponentELO']
    odds = ['TeamWinOdds', 'OpponentWinOdds', 'DrawOdds','O2.5', 'U2.5']
    #print(preds_, df_rolling.filter(like='FIFA').columns.T)
    preds_ = preds_ + odds + list(df_rolling.filter(like='FIFA').columns)
    targets = ['WinLoseTarget', 'OU1.5Target', 'OU2.5Target', 
    'OU3.5Target', 'WinTarget', 'DrawTarget', 'LossTarget']
    targets = ['OU1.5Target', 'OU2.5Target', 
    'OU3.5Target', 'WinTarget', 'DrawTarget', 'LossTarget']
    #targets = ['OU2.5Target']
    #targets = ['WinLoseTarget', 'Target', 'WinTarget', 'DrawTarget', 'LossTarget']

    for target in targets:
        combined = train(date, target, preds_)
