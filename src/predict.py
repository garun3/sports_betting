import pandas as pd
import os
import pickle
from datetime import date as d, timedelta
import numpy as np
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

def format_preds():
    league = config.LEAGUE#'PremierLeague'
    leagues = ['PremierLeague', 'LaLiga', 'SerieA'] + [league]
    leagues = [league]
    data_locs = ['Testing', 'Predictions']
    
    for data in data_locs:
        test = pd.DataFrame()
        new = pd.DataFrame()
        merged = pd.DataFrame()
        #merged1 = pd.read_csv(f'{config.PATH}/../{data}/{league}/{league}_OU2.5Target.csv')
        test1 = pd.read_csv(f'{config.PATH}/../{data}/{league}/{league}_OU2.5Target.csv')
        new1 = pd.read_csv(f'{config.PATH}/{config.CACHE_PATH}/new.csv')
        test = pd.concat([test, test1])
        new = pd.concat([new, new1])
        #merged = pd.concat([merged, merged1])

        merged = test.merge(new[['Date','Team', 'Opponent', 'O2.5', 'U2.5', 'TeamWinOdds', 'DrawOdds', 'OpponentWinOdds']], left_on=['Date','Team_x', 'Opponent_x'], right_on=['Date','Team', 'Opponent'])

        merged['AmericanO2.5'] = np.where(merged['O2.5'] >= 2, round((merged['O2.5'] - 1) * 100), round(-100 / (merged['O2.5'] - 1)))
        merged['AmericanU2.5'] = np.where(merged['U2.5'] >= 2, round((merged['U2.5'] - 1) * 100), round(-100 / (merged['U2.5'] - 1)))

        merged['avgOverPred'] = (merged['class_1_x'] + merged['class_1_y']) / 2
        merged['avgUnderPred'] = (merged['class_0_x'] + merged['class_0_y']) / 2
        merged['impliedOverOdds'] = (1 / merged['O2.5'])
        merged['impliedUnderOdds'] = (1 / merged['U2.5'])

        betAmt = 10
        merged['EVO2.5'] = (merged['avgOverPred'] * (merged['O2.5'] * betAmt - betAmt)) - (1 - merged['avgOverPred']) * betAmt 
        merged['EVU2.5'] = (merged['avgUnderPred'] * (merged['U2.5'] * betAmt - betAmt)) - (1 - merged['avgUnderPred']) * betAmt 
        merged = merged.sort_values('Date')

        merged['Positive Value Over Bet'] = (merged['EVO2.5'] > 0) & (merged['avgOverPred'] > 0.6)
        merged['Positive Value Under Bet'] = (merged['EVU2.5'] > 0) & (merged['avgUnderPred'] > 0.6)
        merged['Positive Value Bet'] = (merged['EVU2.5'] > 0) & (merged['avgUnderPred'] > 0.6)| (merged['EVO2.5'] > 0) & (merged['avgOverPred'] > 0.6)
        merged = merged.drop(columns=['GameID_x', 'GameID_y','Team_x', 'Team_y', 'Opponent_x', 'Opponent_y'])
        merged.to_csv(f'{config.PATH}/../{data}/{league}/{league}_OU2.5Target_.csv', index=False)
    
def merge_league_preds():
    leagues = ['PremierLeague', 'LaLiga', 'SerieA']
    preds = pd.DataFrame()
    date = str(d.today())

    for league in leagues:
        preds1 = pd.read_csv(f'{config.PATH}/../Predictions/{league}/{league}_OU2.5Target_.csv')
        preds = pd.concat([preds, preds1])
    preds = preds.drop(columns=list(preds.filter(like='Unnamed').columns), axis = 1)
    
    preds.sort_values('Date').to_csv(f'{config.PATH}/../Predictions/daily/{date}_OU2.5Target_Predictions.csv', index=False)



if __name__ == "__main__":
    format_preds()
    merge_league_preds()
    '''
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
    '''