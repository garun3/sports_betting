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

import config
from gather_data import get_elo_data, get_current_season
#from gather_data import get_elo_data, 

class MissingDict(dict):
    __missing__ = lambda self, key: key

mapping = MissingDict(**config.TEAM_DIC)

def rename_cols(df):
    c1 = df.filter(like='HomeTeam').columns

    c2 = c1.str.replace('HomeTeam', 'Team')
    df = df.rename(columns={**dict(zip(c1, c2)), **dict(zip(c2, c1))})
    c1 = df.filter(like='AwayTeam').columns

    c2 = c1.str.replace('AwayTeam', 'Opponent')

    df = df.rename(columns={**dict(zip(c1, c2)), **dict(zip(c2, c1))})
    df['Team'] = df['Team'].map(mapping)
    df['Opponent'] = df['Opponent'].map(mapping)

    df['Venue'] = 'Home'
    return df

def home_away(df):
    if 'Team' not in df.columns:
        df = rename_cols(df)
    if 'GameID' not in df.columns:
        df['GameID'] = range(df.shape[0])

    c1 = df.filter(like='Team').columns
    c2 = c1.str.replace('Team', 'Opponent')
    
    swap = df.rename(columns={**dict(zip(c1, c2)), **dict(zip(c2, c1))})
    swap['Venue'] = 'Away'
    df['Result'] = np.where(df['Result'] == 'H', 'W', np.where(df['Result'] == 'A', 'L', df['Result']))
    swap['Result'] = np.where(swap['Result'] == 'H', 'L', np.where(swap['Result'] == 'A', 'W', swap['Result']))
    #swap['Result'] = np.where(swap['Result'] == 'H', 'A', np.where(swap['Result'] == 'A', 'H', 'D'))
    #print(swap)
    df = pd.concat([df,swap])
    
    #df['Result'] = np.where(df['Result'] != 'W', 'L', df['Result'])
    #df = df.rename(columns={'Team1':'Team', 'Team2':'Opponent'})
    df = df.sort_values(by='GameID').reset_index(drop=True)
    return df

#@cache_to_csv(config.CACHE_PATH + "processed1.csv", refresh_time=10)#config.REFRESH_TIME)
def preprocess(total_df):    
    try:
        total_df['Date'] = pd.to_datetime(total_df['Date'], dayfirst=True, format='mixed')
    except:
        total_df['Date'] = pd.to_datetime(total_df['Date'], dayfirst=True, format='mixed')
    
    total_df = total_df.sort_values(by='Date')
    total_df['GameID'] = range(len(total_df))

    df = total_df
    #return df
    #return df.filter(like='>2.5')#.isna().sum()
    df = total_df[['GameID', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'BbAv>2.5', 'BbAv<2.5', 'AvgC>2.5', 'AvgC<2.5', 'B365H', 'B365D', 'B365A', 'AvgCH', 'AvgCD', 'AvgCA', 'B365>2.5', 'B365<2.5']]
    
    #df['O2.5'] = np.where(df['AvgC>2.5'].isna(),np.where(~df['BbAv>2.5'].isna(),df['BbAv>2.5'],df['B365>2.5']),df['AvgC>2.5'])
    #df['O2.5'] = df['B365>2.5']
    #df['U2.5'] = df['B365<2.5']
    df['O2.5'] = np.where(df['BbAv>2.5'].isna(),df['B365>2.5'],df['BbAv>2.5'])
    df['U2.5'] = np.where(df['AvgC<2.5'].isna(),np.where(~df['BbAv<2.5'].isna(),df['BbAv<2.5'],df['B365<2.5']),df['AvgC<2.5'])
    #df['U2.5'] = np.where(df['BbAv<2.5'].isna(),df['B365<2.5'],df['BbAv<2.5'])

    #['O2.5'].isna().sum())
    
    df['HomeTeamWinOdds'] = df['B365H']#np.where(df['AvgCH'].isna(),df['BbAvH'],df['AvgCH'])
    df['AwayTeamWinOdds'] = df['B365A']#np.where(df['AvgCA'].isna(),df['BbAvA'],df['AvgCA'])
    df['DrawOdds'] = df['B365D']#np.where(df['AvgCD'].isna(),df['BbAvD'],df['AvgCD'])
    
    df.columns = ['GameID', 'Date', 'HomeTeam', 'AwayTeam', 'HomeTeamGoals', 'AwayTeamGoals', 'Result', 'HomeTeamShots', 'AwayTeamShots', 'HomeTeamShotsonTarget', 'AwayTeamShotsonTarget', 'HomeTeamFouls', 'AwayTeamFouls', 'HomeTeamCorners', 'AwayTeamCorners', 'HomeTeamYellowCards', 'AwayTeamYellowCards', 'HomeTeamRedCards', 'AwayTeamRedCards', 'BbO2.5', 'BbU2.5', 'AvgO2.5', 'AvgU2.5','BbHome', 'BbDraw', 'BbAway', 'AvgH', 'AvgD', 'AvgA', 'B365O', 'B365U', 'O2.5', 'U2.5', 'HomeTeamWinOdds',  'AwayTeamWinOdds', 'DrawOdds']

    df = df[['GameID', 'Date', 'HomeTeam', 'AwayTeam', 'HomeTeamGoals', 'AwayTeamGoals', 'Result', 'HomeTeamShots', 'AwayTeamShots', 'HomeTeamShotsonTarget', 'AwayTeamShotsonTarget', 'HomeTeamFouls', 'AwayTeamFouls', 'HomeTeamCorners', 'AwayTeamCorners', 'HomeTeamYellowCards', 'AwayTeamYellowCards', 'HomeTeamRedCards', 'AwayTeamRedCards', 'O2.5', 'U2.5', 'HomeTeamWinOdds', 'DrawOdds', 'AwayTeamWinOdds']]
    df = df.sort_values(by='Date')

    df = home_away(df)
    return df

def create_predictors(df):
    print(df.columns)
    df['VenueCode'] = df['Venue'].astype('category').cat.codes
    df['OpponentCode'] = df['Opponent'].astype('category').cat.codes
    df['DayCode'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['OU1.5Target'] = (df['TotalGoals'] > 1.5).astype('int')
    df['OU2.5Target'] = (df['TotalGoals'] > 2.5).astype('int')
    df['OU3.5Target'] = (df['TotalGoals'] > 3.5).astype('int')
    df['WinLoseTarget'] = df['Result'].astype('category').cat.codes #(new['Result'] == 'W').astype('int')
    df['WinTarget'] = (df['Result'] == 'W').astype('int')#new['Result'].astype('category').cat.codes
    df['DrawTarget'] = (df['Result'] == 'D').astype('int')
    df['LossTarget'] = (df['Result'] == 'L').astype('int')
    df['OUCornersTarget'] = (df['TotalCorners'] > 9.5).astype('int')
    return df.reset_index(drop=True)

def rolling_averages(group, cols, new_cols):
    group = group.sort_values(by='Date')
    rolling_stats = group[cols].rolling(3, min_periods=2, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

def create_rolling(merged_df, cols):
    
    new_cols = [f'{c}Rolling' for c in cols]
    df_rolling = merged_df.groupby('Team').apply(lambda x: rolling_averages(x, cols, new_cols))

    df_rolling = df_rolling.droplevel('Team')
    df_rolling = df_rolling.sort_values('GameID')
    df_rolling.index = range(df_rolling.shape[0])
    return df_rolling, new_cols

def calculate_difs(df, cols, rolling=True):
    if rolling:
        new_cols = [f'{c}Rolling_Dif' for c in cols]
        for i,c in enumerate(cols):
            df[new_cols[i]] = df[f'Team{c}Rolling'] - df[f'Opponent{c}Rolling']
    else:
        new_cols = [f'{c}Dif' for c in cols]
        for i,c in enumerate(cols):
            df[new_cols[i]] = df[f'Team{c}'] - df[f'Opponent{c}']
    return df, new_cols

def add_xg(df):
    path = f'{config.PATH}/{config.CACHE_PATH}/xG.csv'
    print(path)
    xG = pd.read_csv(path)
    xG = home_away(xG)
    print(xG['Team'].unique(), df['Team'].unique())
    '''
    print teams with different names
    '''
    li1 = xG['Team'].unique()
    s = set(df['Team'].unique())
    print('teams with different names:', [x for x in li1 if x not in s])

    df['Date'] = pd.to_datetime(df['Date'])
    xG['Date'] = pd.to_datetime(xG['Date'])
    df = df.merge(xG[['Date', 'Team', 'TeamxG', 'OpponentxG']], on=['Date', 'Team'], how='left')
    return df

def add_ou(df):
    path = f'{config.PATH}/{config.CACHE_PATH}/current_odds.csv'
    print(path)
    ou = pd.read_csv(path)
    ou = rename_cols(ou)
    li1 = ou['Team'].unique()
    s = set(df['Team'].unique())
    print('teams with different names:', [x for x in li1 if x not in s])
    #return ou
    df['Date'] = pd.to_datetime(df['Date'])
    ou['Date'] = pd.to_datetime(ou['Date'])
    df = df.merge(ou.drop(columns='Venue'), on=['Date', 'Team', 'Opponent'], how='left')
    return df

def calculate_totals(df, cols):
    new_cols = [f'Total{col}' for col in cols]
    for i, new_col in enumerate(new_cols):
        df[new_col] = df[f'Team{cols[i]}'] + df[f'Opponent{cols[i]}']   
    return df, new_cols

def create_stats(df):
    if config.XG:
        total_cols = ['Goals', 'xG', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards', 'ELO']
        total_cols1 = ['Goals', 'xG', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards']
        ind_cols = ['TeamGoals', 'OpponentGoals', 'TeamxG', 'OpponentxG', 'TeamShots', 'OpponentShots', 'TeamFouls', 'OpponentFouls','TeamShotsonTarget', 'OpponentShotsonTarget', 'TeamCorners', 'OpponentCorners', 'TeamYellowCards', 'OpponentYellowCards','TeamRedCards','OpponentRedCards']
    else:
        total_cols = ['Goals', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards', 'ELO']
        total_cols1 = ['Goals', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards']
        ind_cols = ['TeamGoals', 'OpponentGoals', 'TeamShots', 'OpponentShots', 'TeamFouls', 'OpponentFouls','TeamShotsonTarget', 'OpponentShotsonTarget', 'TeamCorners', 'OpponentCorners', 'TeamYellowCards', 'OpponentYellowCards','TeamRedCards','OpponentRedCards']
    pred_cols = []
    
    df = get_elo_data(df)
    if config.XG:
        df = add_xg(df)
    print(df['TeamShots'].tail())
    df, new_cols = calculate_totals(df, total_cols)
    df, new_cols = create_rolling(df, new_cols)
    pred_cols.extend(new_cols)
    df, new_cols = create_rolling(df, ind_cols)
    print(df['TeamShots'].tail())
    pred_cols.extend(new_cols)

    df, dif_cols = calculate_difs(df, total_cols1, rolling=True)

    pred_cols.extend(dif_cols)
    return df, pred_cols

def prep_data(rolling, date):
    if config.XG:
        total_cols = ['Goals', 'xG', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards', 'ELO']
        total_cols1 = ['Goals', 'xG', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards']
        ind_cols = ['TeamGoals', 'OpponentGoals', 'TeamxG', 'OpponentxG', 'TeamShots', 'OpponentShots', 'TeamFouls', 'OpponentFouls','TeamShotsonTarget', 'OpponentShotsonTarget', 'TeamCorners', 'OpponentCorners', 'TeamYellowCards', 'OpponentYellowCards','TeamRedCards','OpponentRedCards']
    else:
        total_cols = ['Goals', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards', 'ELO']
        total_cols1 = ['Goals', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards']
        ind_cols = ['TeamGoals', 'OpponentGoals', 'TeamShots', 'OpponentShots', 'TeamFouls', 'OpponentFouls','TeamShotsonTarget', 'OpponentShotsonTarget', 'TeamCorners', 'OpponentCorners', 'TeamYellowCards', 'OpponentYellowCards','TeamRedCards','OpponentRedCards']
    #rolling = pd.read_csv(f'cache/LaLiga/rolling.csv')
    elo = pd.read_csv(f'{config.PATH}/{config.CACHE_PATH}/elo.csv')
    schedule = pd.read_csv(f'{config.PATH}/{config.CACHE_PATH}/schedule.csv')#get_schedule()#pd.read_csv(f'cache/{self.league}/schedule.csv')
    #print(rolling['Date'])
    recent_rolling = rolling.groupby(['Team']).last().reset_index()
    #schedule.loc[(schedule['Date'] == '23/09/23') | (schedule['Date'] == '24/09/23')]
    #print(recent_rolling.Team)
    #for date in dates:
    #print(schedule.Date)
    
    sched = schedule.loc[schedule['Date'] >= date]#(schedule['Date'] == date)]])
    #print(schedule['Date'], sched['Date'],date)
    #print(sched.shape, 'hi')
    sched['Result'] = -1
    sched = rename_cols(sched)
    #print(recent_rolling)
    #print(sched.shape, 'hi', recent_rolling.columns, sched.columns)
    #xG = pd.read_csv(f'cache/{league}/xG.csv')

    
    #sched = self.add_xg(sched)
    #print(sched.columns, 'columns')
    sched['Date'] = pd.to_datetime(sched['Date'])
    sched = sched.merge(recent_rolling[['Team', 'TeamELO']], on='Team', how='inner')
    print(sched.head(10)[['Date','Team', 'Opponent']].sort_values('Date'))
    #print(sched.columns, 'columns')
    #print(sched.shape)
    sched = sched.merge(recent_rolling[['Opponent', 'OpponentELO']], on='Opponent', how='inner')
    
    #return sched.merge(xG[['Date', 'Team', 'TeamxG', 'OpponentxG']], on=['Date', 'Team'])
    
    sched = sched.sort_values('Date').drop_duplicates(['Date', 'Team'])
    print(sched.head(10)[['Date','Team', 'Opponent']])
    
    if config.XG:
        sched = add_xg(sched)
    sched = add_ou(sched)
    print(sched.head(10))
    sched = home_away(sched)
    

    sched, _ = calculate_totals(sched, ['ELO'])
    sched, _ = calculate_difs(sched, ['ELO'], rolling=False)
    
    sched = sched.sort_values('Date')
    #sched, _ = self.create_rolling(sched, ['TotalELO'])
    #return sched
    print(sched['Date'].unique())
    #return sched
    #print(sched, 'team')
    #print(sched,' hey')
    #sched['ELODif'] = sched['TeamELO'] - sched['OpponentELO']
    #sched, cols = self.create_stats(sched)
    #return sched
    
    #sched['Date'] = pd.to_datetime(sched['Date'], dayfirst=True).dt.date
    #print(sched['Team'], 'team')
    #print(sched.shape)#['Date'], 'hi')
    cols = ['Date', 'Team', 'Opponent', 'TeamELO', 'OpponentELO', 'ELODif', 'Venue', "GameID"]
    #return sched[sched.columns.intersection(elo.columns)]
    new = pd.concat([rolling, sched[sched.columns.intersection(rolling.columns)]], axis=0).reset_index(drop=True)
    #return new
    print(new['Date'].unique())
    #['Date'].unique(),'hi')
    #new['TotalGoals'] = new['TeamGoals'] + new['OpponentGoals']
    new['Date'] = pd.to_datetime(new['Date'], dayfirst=True).dt.date
    #total_cols = ['Goals', 'xG', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards', 'ELO']
    #new[new['Date'] > date]['ELO']
    #new = self.calculate_totals(new)
    #return new
    #new = self.create_predictors(new)
    new, new_cols = calculate_totals(new, total_cols)
    new, new_cols = create_rolling(new, new_cols)
    new, new_cols = create_rolling(new, ind_cols)
    new, dif_cols = calculate_difs(new, total_cols1)
    new = create_predictors(new)

    #numeric_cols = ['Shots', 'ShotsonTarget', 'Fouls', 'Corners', 'YellowCards', 'RedCards']
    #cols = ['Goals','Shots', 'ShotsonTarget', 'Corners']
    #dif_cols = 
    #print(new.columns, 'h')
    #new, dif_cols = self.calculate_difs(new, cols)
    #new['WinLoseTarget'] = new['Result'].astype('category').cat.codes #(new['Result'] == 'W').astype('int')
    #new['WinTarget'] = (new['Result'] == 'W')#new['Result'].astype('category').cat.codes
    #new['DrawTarget'] = (new['Result'] == 'D')
    #new['LossTarget'] = (new['Result'] == 'L')
    #print(new['Date'])
    new.to_csv(f'{config.PATH}/{config.CACHE_PATH}/new.csv')
    '''
    #rolling = pd.read_csv(f'cache/LaLiga/rolling.csv')
    elo = pd.read_csv(f'cache/{self.league}/elo.csv')
    schedule = self.get_schedule()#pd.read_csv(f'cache/{self.league}/schedule.csv')
    #print(rolling['Date'])
    recent_rolling = rolling.groupby(['Team']).last().reset_index()
    #schedule.loc[(schedule['Date'] == '23/09/23') | (schedule['Date'] == '24/09/23')]
    #print(recent_rolling.Team)
    #for date in dates:
    #print(schedule.Date)
    
    sched = schedule.loc[schedule['Date'] >= date]#(schedule['Date'] == date)]])
    print(schedule['Date'], sched['Date'],date)
    #print(sched.shape, 'hi')
    sched['Result'] = -1
    sched = self.rename_cols(sched)
    #print(recent_rolling)
    #print(sched.shape, 'hi', recent_rolling.columns, sched.columns)
    #xG = pd.read_csv(f'cache/{league}/xG.csv')
    if self.xg:
        sched = self.add_xg(sched)#sched.merge(xG[['Date', 'Team', 'TeamxG', 'OpponentxG']], on=['Date', 'Team'])
    sched = sched.merge(recent_rolling[['Team', 'TeamELO']], on='Team', how='inner')
    #print(sched.shape)
    sched = sched.merge(recent_rolling[['Opponent', 'OpponentELO']], on='Opponent', how='inner')
    sched = sched.sort_values('Date').drop_duplicates(['Date', 'Team'])
    sched = self.home_away(sched)
    #print(sched, 'team')
    print(sched,' hey')
    sched['ELODif'] = sched['TeamELO'] - sched['OpponentELO']
    #sched['Date'] = pd.to_datetime(sched['Date'], dayfirst=True).dt.date
    #print(sched['Team'], 'team')
    #print(sched.shape)#['Date'], 'hi')
    new = pd.concat([elo, sched[['Date', 'Team', 'Opponent', 'TeamELO', 'OpponentELO', 'ELODif', 'Venue', "GameID"]]])
    print(new)#['Date'].unique(),'hi')
    #return new

    new['Date'] = pd.to_datetime(new['Date'], dayfirst=True).dt.date
    
    new = self.create_predictors(new)
    new = self.create_rolling(new)
    print(new['Date'],date)
    numeric_cols = ['Shots', 'ShotsonTarget', 'Fouls', 'Corners', 'YellowCards', 'RedCards']
    cols = ['Goals','Shots', 'ShotsonTarget', 'Corners']
    #print(new.columns, 'h')
    new, dif_cols = self.calculate_difs(new, cols)
    #new['WinLoseTarget'] = new['Result'].astype('category').cat.codes #(new['Result'] == 'W').astype('int')
    #new['WinTarget'] = (new['Result'] == 'W')#new['Result'].astype('category').cat.codes
    #new['DrawTarget'] = (new['Result'] == 'D')
    #new['LossTarget'] = (new['Result'] == 'L')
    #print(new['Date'])
    new.to_csv(f'cache/{self.league}/new.csv')
    return dif_cols
    '''

if __name__ == "__main__":
    print('preprocess')
    league_df = pd.read_csv(f'{config.PATH}/{config.CACHE_PATH}/league.csv')
    schedule_df = pd.read_csv(f'{config.PATH}/{config.CACHE_PATH}/current_season.csv')
    print(schedule_df.head())
    new_df = pd.concat([league_df, schedule_df])
    print(new_df.tail(10))
    df1 = preprocess(new_df)
    #df1 = func.add_xg(df1)
    # creates totals columns and adds rolling totals and rolling team stats
    df, rolling_cols = create_stats(df1)
    df_rolling = create_predictors(df)
    df_rolling.to_csv(f'{config.PATH}/{config.CACHE_PATH}/rolling.csv')
    date = str(d.today())
    prep_data(df_rolling, date)