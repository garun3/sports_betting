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
import requests
from flatten_json import flatten
import json

import config

class MissingDict(dict):
    __missing__ = lambda self, key: key

mapping = MissingDict(**config.TEAM_DIC)

@cache_to_csv(config.CACHE_PATH + 'league.csv', refresh_time=604800)#config.REFRESH_TIME)
def get_league_data():
    directory = f'{config.PATH}/../Data/{config.LEAGUE}'
    total_df = pd.DataFrame()
    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename[-3:] == 'csv':
            print(f)
            df = pd.read_csv(f, encoding='unicode_escape')
            total_df = pd.concat([total_df,df], join='outer')
    return total_df



@cache_to_csv(config.CACHE_PATH + 'schedule.csv', refresh_time=config.REFRESH_TIME)
def get_schedule():
    
    fbref = sd.FBref(config.LEAGUE_DIC[config.LEAGUE], '2023')
    sched23 = fbref.read_schedule()

    sched23 = sched23.droplevel('league')
    sched23 = sched23.droplevel('season')

    sched23 = sched23.rename(columns={'date':'Date', 'home_team':'HomeTeam', 'away_team':'AwayTeam', 'referee': 'Referee'})
    sched23['Date'] = pd.to_datetime(sched23['Date'], yearfirst=True).dt.strftime('%Y-%m-%d')
    sched23[['BbAv>2.5', 'BbAv<2.5', 'BbAvH', 'BbAvD', 'BbAvA']] = None
    sched23['Venue'] = 'Home'
    sched23 = sched23.reset_index(drop=True)
    return sched23

@cache_to_csv(config.CACHE_PATH + "current_season.csv", refresh_time=config.REFRESH_TIME)
def get_current_season():
    df = pd.DataFrame()
    #years =  ['0102', '0203','0304','0405','0506','0607','0708','0809','0910','1011','1112','1213','1314','1415','1516','1617','1718','1819','1920','2021','2122','2223','2323']
    years = ['2324']

    for year in years: 
        history = sd.MatchHistory(config.LEAGUE_DIC[config.LEAGUE], year)
        sched = history.read_games()
        df = pd.concat([df, sched])

    sched23 = df

    sched23 = sched23.droplevel('league')
    sched23 = sched23.droplevel('season')

    sched23 = sched23.rename(columns={'date':'Date', 'home_team':'HomeTeam', 'away_team':'AwayTeam', 'referee': 'Referee'})
    sched23['Date'] = pd.to_datetime(sched23['Date'], yearfirst=True).dt.strftime('%d-%m-%y')
    
    #sched23[['BbAv>2.5', 'BbAv<2.5', 'BbAvH', 'BbAvD', 'BbAvA']] = None
    sched23['HomeTeam'] = sched23['HomeTeam'].map(mapping)
    sched23['AwayTeam'] = sched23['AwayTeam'].map(mapping)
    sched23 = sched23.reset_index(drop=True)
    return sched23

@cache_to_csv(config.CACHE_PATH + "elo_raw.csv", refresh_time=86400)#config.REFRESH_TIME)
def get_elo_data(full_df):
    clubelo = sd.ClubElo(config.LEAGUE_DIC[config.LEAGUE])
    missing = []
    # create elo df
    elo_df = pd.DataFrame()
    #print(full_df['Team'].unique())
    for club in full_df['Team'].unique():
        print('hi', club)
        try:
            elo = clubelo.read_team_history(club)
            elo_df = pd.concat([elo_df,elo])
        except:
            missing.append(club)
    print(missing)
    elo_df = elo_df.reset_index()
    elo_df = elo_df[elo_df['from'] > '2000'] 
    out = add_elo(elo_df, full_df)
    return out

@cache_to_csv(config.CACHE_PATH + "elo.csv", refresh_time=config.REFRESH_TIME)
def add_elo(elo_df, full_df):
    #elo_df = pd.read_csv(config.CACHE_PATH + "elo_raw.csv")
    out = full_df.merge(elo_df[['team', 'elo', 'from', 'to']], how='left', left_on=['Team'], right_on=['team']) 
    out = out.query('Date.between(`from`, `to`)')
    out = out.merge(elo_df[['team', 'elo', 'from', 'to']], how='left', left_on=['Opponent'], right_on=['team']) 
    out = out.query('Date.between(`from_y`, `to_y`)')
    out = out.reset_index(drop=True)
    out['ELODif'] = out['elo_x'] - out['elo_y']
    out = out.rename(columns={'elo_x':'TeamELO', 'elo_y':'OpponentELO'})
    out = out.drop(columns=['from_x', 'from_y', 'to_x', 'to_y', 'team_x', 'team_y'])
    return out

@cache_to_csv(config.CACHE_PATH + "understat.csv", refresh_time=259200)
def get_understat():
    league_dic = {'PremierLeague': 'EPL',
    'LaLiga': 'La_Liga',
    'SerieA': 'Serie_A'}
    understat = UnderstatClient()
    years = range(2014,2024)
    teams = set()
    for year in years:
        t = understat.league(league_dic[config.LEAGUE]).get_team_data(str(year))
        df = pd.DataFrame(t)
        ts = df.reset_index().iloc[1].unique()[1:]
        y = set(ts)
        teams.update(y)
    #print(teams)
    understat = UnderstatClient()
    df = pd.DataFrame()
    missing = {}
    #print(teams)
    for team in teams:
        t = understat.team(team=team)
        print(team)
        for year in years:
            print(year)
            try:
                y = pd.json_normalize(t.get_match_data(str(year)))
                df = pd.concat([df, y])
            except:
                missing[year] = missing.get(year, []) + [team]
    print(f'missing understat teams: {missing}')
    return df

@cache_to_csv(config.CACHE_PATH + "xG.csv", refresh_time=config.REFRESH_TIME)
def get_xg():  
    df1 = get_understat()
    df1 = df1.sort_values('datetime')
    df1 = df1.rename(columns={'h.title':'HomeTeam','a.title':'AwayTeam', 'datetime': 'Date',
    'xG.h': 'HomeTeamxG', 'xG.a': 'AwayTeamxG',
    'result': 'Result'})

    df1['Date'] = pd.to_datetime(df1['Date']).dt.strftime('%Y-%m-%d')

    df1['Result'] = df1['Result'].str.upper()
    df1 = df1.drop_duplicates(['id'])#['Date', 'HomeTeam', 'AwayTeam'])
    #df1.to_csv(f'cache/{self.league}/xG.csv')
    
    return df1

@cache_to_csv(config.CACHE_PATH + "current_odds.csv", refresh_time=86400)#config.REFRESH_TIME)
def get_current_odds():
    API_KEY = config.API_KEY

    SPORT = 'upcoming' 
    REGIONS = 'uk' 
    MARKETS = 'totals,h2h' # h2h | spreads | totals
    BOOKMAKERS = 'williamhill'

    league_dic = {'PremierLeague':'soccer_epl',
    'LaLiga':'soccer_spain_la_liga',
    'SerieA': 'soccer_italy_serie_a'}
    SPORT = league_dic[config.LEAGUE]
    odds_response = requests.get(f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds', params={
        'api_key': API_KEY,
        'regions': REGIONS,
        'bookmakers': BOOKMAKERS,
        'markets': MARKETS,
    })

    if odds_response.status_code != 200:
        print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')

    else:
        odds_json = odds_response.json()
        print('Number of events:', len(odds_json))
        #print(odds_json)

        # Check the usage quota
        print('Remaining requests', odds_response.headers['x-requests-remaining'])
        print('Used requests', odds_response.headers['x-requests-used'])

    dict_flat = (flatten(record, '.') for record in odds_json)
    df = pd.DataFrame(dict_flat)
    df['Date'] = pd.to_datetime(df['commence_time']).dt.date#.filter(like='sites')
    df = df.rename(columns={'home_team': 'HomeTeam',
                            'away_team': 'AwayTeam'})
    df['HomeTeamWinOdds'] = np.where(df['bookmakers.0.markets.0.outcomes.0.name'] == df['HomeTeam'], df['bookmakers.0.markets.0.outcomes.0.price'], df['bookmakers.0.markets.0.outcomes.1.price'])
    df['AwayTeamWinOdds'] = np.where(df['bookmakers.0.markets.0.outcomes.0.name'] == df['AwayTeam'], df['bookmakers.0.markets.0.outcomes.0.price'], df['bookmakers.0.markets.0.outcomes.1.price'])
    df['DrawOdds'] = df['bookmakers.0.markets.0.outcomes.2.price']
    df['O2.5'] = np.where(df['bookmakers.0.markets.1.outcomes.0.name'] == 'Over', df['bookmakers.0.markets.1.outcomes.0.price'], df['bookmakers.0.markets.1.outcomes.1.price'])
    df['U2.5'] = np.where(df['bookmakers.0.markets.1.outcomes.0.name'] == 'Under', df['bookmakers.0.markets.1.outcomes.0.price'], df['bookmakers.0.markets.1.outcomes.1.price'])

    cols = ['Date','HomeTeam', 'AwayTeam','O2.5','U2.5', 'HomeTeamWinOdds', 'AwayTeamWinOdds', 'DrawOdds']
    df_odds = df[cols]	
    return df_odds

@cache_to_csv(config.CACHE_PATH + "fifa.csv", refresh_time=259200)
def update_fifa():
    df = pd.read_csv(f'{config.PATH}/{config.CACHE_PATH}/fifa.csv')
    fifa = sd.SoFIFA(leagues=config.LEAGUE, versions='latest')

    new = fifa.read_team_ratings()
    new_dates = new['update']
    if new_dates[0] not in df['update']:
        df = pd.concat([df, new])
    return df
    #print(df.tail(20))
    #df.to_csv(f'{config.PATH}/{config.CACHE_PATH}/fifa.csv')


if __name__ == "__main__":    
    print('getting league')
    prem_df = get_league_data()
    print('getting schedule')
    sched23 = get_current_season()
    schedule = get_schedule()
    if config.XG:
        get_xg()
    get_current_odds()
    
    update_fifa()
