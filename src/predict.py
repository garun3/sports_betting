import pandas as pd
import os
import pickle
from datetime import date as d, timedelta
import config

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
    X_train, y_train, X_test, y_test = split_data(new_df, [], predictors, pred_type)
    

    preds = model.predict(pred[predictors])
    return preds, pred

if __name__ == "__main__":
    date = str(d.today())
    if config.XG:
        predictors = ['TotalShotsRolling', 'TeamGoalsRolling', 'TeamFoulsRolling', 'GoalsRolling_Dif', 'ShotsonTargetRolling_Dif', 'OpponentShotsonTargetRolling', 'TotalELORolling', 'xGRolling_Dif', 'ShotsRolling_Dif', 'YellowCardsRolling_Dif', 'TeamRedCardsRolling', 'TotalRedCardsRolling', 'OpponentYellowCardsRolling', 'ELODif', 'TotalFoulsRolling', 'TeamShotsonTargetRolling', 'OpponentCornersRolling', 'TotalCornersRolling', 'FoulsRolling_Dif', 'OpponentGoalsRolling', 'RedCardsRolling_Dif', 'TotalGoalsRolling', 'TeamShotsRolling', 'CornersRolling_Dif', 'TeamCornersRolling', 'OpponentCode', 'TotalYellowCardsRolling', 'VenueCode', 'OpponentRedCardsRolling', 'OpponentShotsRolling', 'TeamYellowCardsRolling', 'TotalShotsonTargetRolling', 'DayCode', 'OpponentFoulsRolling']
    else:
        predictors = ['TotalShotsRolling', 'TeamGoalsRolling', 'TeamFoulsRolling', 'GoalsRolling_Dif', 'ShotsonTargetRolling_Dif', 'OpponentShotsonTargetRolling', 'TotalELORolling', 'ShotsRolling_Dif', 'YellowCardsRolling_Dif', 'TeamRedCardsRolling', 'TotalRedCardsRolling', 'OpponentYellowCardsRolling', 'ELODif', 'TotalFoulsRolling', 'TeamShotsonTargetRolling', 'OpponentCornersRolling', 'TotalCornersRolling', 'FoulsRolling_Dif', 'OpponentGoalsRolling', 'RedCardsRolling_Dif', 'TotalGoalsRolling', 'TeamShotsRolling', 'CornersRolling_Dif', 'TeamCornersRolling', 'OpponentCode', 'TotalYellowCardsRolling', 'VenueCode', 'OpponentRedCardsRolling', 'OpponentShotsRolling', 'TeamYellowCardsRolling', 'TotalShotsonTargetRolling', 'DayCode', 'OpponentFoulsRolling']

    preds_ = predictors + ['TeamELO', 'OpponentELO']
    odds = ['TeamWinOdds', 'OpponentWinOdds', 'DrawOdds','O2.5', 'U2.5']
    preds_ = preds_ + odds
    targets = ['WinLoseTarget', 'OU1.5Target', 'OU2.5Target', 
    'OU3.5Target', 'WinTarget', 'DrawTarget', 'LossTarget']
    #targets = ['OU2.5Target']
    #targets = ['WinLoseTarget', 'Target', 'WinTarget', 'DrawTarget', 'LossTarget']


    for target in targets:
        preds, c = predict(predictors, target)

        c['preds'] = preds
        table = c.sort_index()[['GameID','Date','Team', 'Opponent', 'preds']].merge(c[['GameID','Date','Team', 'Opponent', 'preds']], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])
        table.to_csv(f'{config.PATH}/../Predictions/{config.LEAGUE}/{config.LEAGUE}_{target}.csv')
         
        #c['preds'] = preds
        #table = c.sort_index()[['GameID','Date','Team', 'Opponent', 'preds']].merge(c[['GameID','Date','Team', 'Opponent', 'preds']], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])
        #table.to_csv(f'Predictions/{league}/{league}_{target}.csv')