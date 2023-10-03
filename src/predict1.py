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

    
path = 'cache/PremierLeague/'
#path = 'cache/LaLiga/'
#path = 'cache/SerieA/'
r_time = 600

class Functions:
    def __init__(self, league='PremierLeague', xg=True):
        self.league = league
        self.dic = {'PremierLeague':'ENG-Premier League',
        'LaLiga':'ESP-La Liga',
        'SerieA': 'ITA-Serie A'}
        map_values = {
    'Middlesboro': 'Middlesbrough',
    "Nott'm Forest": 'Forest',
    'Newcastle Utd': 'Newcastle',
    'Luton Town': 'Luton',
    'Manchester City': 'Man City',
    'Manchester Utd': 'Man United',
    'Sheffield Utd': 'Sheffield United',
    "Nott'ham Forest": 'Forest',
    "Nottingham Forest": 'Forest',
    'Manchester United': 'Man United',
    'Wolverhampton Wanderers': 'Wolves',
    'Leicester City': 'Leicester',
    'Newcastle United': 'Newcastle',
    'Queens Park Rangers': 'QPR',
    'West Bromwich Albion': 'West Brom',
    'Vallecano': 'Rayo Vallecano', 
    'Ath Bilbao': 'Bilbao', 
    'Ath Madrid': 'Atletico', 
    'Espanol': 'Espanyol', 
    'Sp Gijon': 'Gijon', 
    'Celta Vigo': 'Celta', 
    'Atlético Madrid': 'Atletico', 
    'Athletic Club': 'Bilbao',
    'Real Sociedad': 'Sociedad',
    'Almería': 'Almeria',
    'Cádiz': 'Cadiz',
    'Alavés': 'Alaves',
    'Hellas Verona': 'Verona',
    'Atletico Madrid': 'Atletico', 
    'Sporting Gijon': 'Gijon',
    'Real Betis': 'Betis',
    'Real Valladolid': 'Valladolid',
    'SD Huesca': 'Huesca'

}
        '''
        'La Coruna', 'Gimnastic', 
        '''
        self.mapping = MissingDict(**map_values)
        self.model = None
        self.filename = f'models/{self.league}_model.sav'
        self.xg = xg
    
    @cache_to_csv(path + 'league.csv', refresh_time=r_time)
    def get_league_data(self):
        directory = f'/Users/Gautham/Projects/Betting/Data/{self.league}'
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

    

    @cache_to_csv(path + 'schedule.csv', refresh_time=r_time)
    def get_schedule(self):
        
        fbref = sd.FBref(self.dic[self.league], '2023')
        sched23 = fbref.read_schedule()

        sched23 = sched23.droplevel('league')
        sched23 = sched23.droplevel('season')

        sched23 = sched23.rename(columns={'date':'Date', 'home_team':'HomeTeam', 'away_team':'AwayTeam', 'referee': 'Referee'})
        sched23['Date'] = pd.to_datetime(sched23['Date'], yearfirst=True).dt.strftime('%Y-%m-%d')
        sched23[['BbAv>2.5', 'BbAv<2.5', 'BbAvH', 'BbAvD', 'BbAvA']] = None
        sched23 = sched23.reset_index(drop=True)
        return sched23
    
    @cache_to_csv(path + "current_season.csv", refresh_time=r_time)
    def get_current_season(self):
        df = pd.DataFrame()
        #years =  ['0102', '0203','0304','0405','0506','0607','0708','0809','0910','1011','1112','1213','1314','1415','1516','1617','1718','1819','1920','2021','2122','2223','2323']
        years = ['2324']

        for year in years: 
            history = sd.MatchHistory(self.dic[self.league], year)
            sched = history.read_games()
            df = pd.concat([df, sched])


        sched23 = df

        sched23 = sched23.droplevel('league')
        sched23 = sched23.droplevel('season')

        sched23 = sched23.rename(columns={'date':'Date', 'home_team':'HomeTeam', 'away_team':'AwayTeam', 'referee': 'Referee'})
        sched23['Date'] = pd.to_datetime(sched23['Date'], yearfirst=True).dt.strftime('%d-%m-%y')
        
        #sched23[['BbAv>2.5', 'BbAv<2.5', 'BbAvH', 'BbAvD', 'BbAvA']] = None
        sched23['HomeTeam'] = sched23['HomeTeam'].map(self.mapping)
        sched23['AwayTeam'] = sched23['AwayTeam'].map(self.mapping)
        sched23 = sched23.reset_index(drop=True)
        return sched23
        
    @cache_to_csv(path + "elo.csv", refresh_time=r_time)
    def get_elo_data(self,full_df):
        clubelo = sd.ClubElo(self.dic[self.league])
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
            #print(elo_df)
        print(missing)
        elo_df = elo_df.reset_index()
        elo_df = elo_df[elo_df['from'] > '2000'] 
        #print(elo_df)

        # add nottingham data
        '''
        elo_df = pd.concat([elo_df, clubelo.read_team_history('Forest')])
        elo_df['team'].replace(['Forest'], "Nott'm Forest", regex=True, inplace=True)
        elo_df = elo_df.reset_index()
        '''
        # merge elo with full df
        out = full_df.merge(elo_df[['team', 'elo', 'from', 'to']], how='left', left_on=['Team'], right_on=['team']) 
        out = out.query('Date.between(`from`, `to`)')
        out = out.merge(elo_df[['team', 'elo', 'from', 'to']], how='left', left_on=['Opponent'], right_on=['team']) 
        out = out.query('Date.between(`from_y`, `to_y`)')
        out = out.reset_index(drop=True)
        out['ELODif'] = out['elo_x'] - out['elo_y']
        out = out.rename(columns={'elo_x':'TeamELO', 'elo_y':'OpponentELO'})
        out = out.drop(columns=['from_x', 'from_y', 'to_x', 'to_y', 'team_x', 'team_y'])
        return out

    @cache_to_csv(path + "understat.csv", refresh_time=160000)
    def get_understat(self):
        league_dic = {'PremierLeague': 'EPL',
        'LaLiga': 'La_Liga',
        'SerieA': 'Serie_A'}
        understat = UnderstatClient()
        years = range(2014,2024)
        teams = set()
        for year in years:
            t = understat.league(league_dic[league]).get_team_data(str(year))
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

    @cache_to_csv(path + "xG.csv", refresh_time=r_time)
    def get_xg(self):  
        df1 = self.get_understat()
        df1 = df1.sort_values('datetime')
        df1 = df1.rename(columns={'h.title':'HomeTeam',                   'a.title':'AwayTeam', 'datetime': 'Date',
        'xG.h': 'HomeTeamxG', 'xG.a': 'AwayTeamxG',
        'result': 'Result'})

        df1['Date'] = pd.to_datetime(df1['Date']).dt.strftime('%Y-%m-%d')

        df1['Result'] = df1['Result'].str.upper()
        df1 = df1.drop_duplicates(['id'])#['Date', 'HomeTeam', 'AwayTeam'])
        df1 = self.home_away(df1)
        #df1.to_csv(f'cache/{self.league}/xG.csv')
        
        return df1

    def rename_cols(self, df):
        c1 = df.filter(like='HomeTeam').columns

        c2 = c1.str.replace('HomeTeam', 'Team')
        df = df.rename(columns={**dict(zip(c1, c2)), **dict(zip(c2, c1))})
        c1 = df.filter(like='AwayTeam').columns

        c2 = c1.str.replace('AwayTeam', 'Opponent')

        df = df.rename(columns={**dict(zip(c1, c2)), **dict(zip(c2, c1))})
        df['Team'] = df['Team'].map(self.mapping)
        df['Opponent'] = df['Opponent'].map(self.mapping)

        df['Venue'] = 'Home'
        return df

    def home_away(self,df):
        if 'Team' not in df.columns:
            df = self.rename_cols(df)
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

    def preprocess(self, total_df):    
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

        df = self.home_away(df)
        return df

    def create_predictors(self,df):
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

    def rolling_averages(self,group, cols, new_cols):
        group = group.sort_values(by='Date')
        rolling_stats = group[cols].rolling(3, min_periods=2, closed='left').mean()
        group[new_cols] = rolling_stats
        group = group.dropna(subset=new_cols)
        return group
    
    def create_rolling(self,merged_df, cols):
        
        new_cols = [f'{c}Rolling' for c in cols]
        df_rolling = merged_df.groupby('Team').apply(lambda x: self.rolling_averages(x, cols, new_cols))

        df_rolling = df_rolling.droplevel('Team')
        df_rolling = df_rolling.sort_values('GameID')
        df_rolling.index = range(df_rolling.shape[0])
        return df_rolling, new_cols

    def calculate_difs(self, df, cols, rolling=True):
        if rolling:
            new_cols = [f'{c}Rolling_Dif' for c in cols]
            for i,c in enumerate(cols):
                df[new_cols[i]] = df[f'Team{c}Rolling'] - df[f'Opponent{c}Rolling']
        else:
            new_cols = [f'{c}Dif' for c in cols]
            for i,c in enumerate(cols):
                df[new_cols[i]] = df[f'Team{c}'] - df[f'Opponent{c}']
        return df, new_cols

    def add_xg(self, df):
        xG = self.get_xg()#pd.read_csv(f'cache/{self.league}/xG.csv')
        '''
        print teams with different names
        '''
        li1 = xG['Team'].unique()
        s = set(df['Team'].unique())
        print('teams with different names:', [x for x in li1 if x not in s])

        df['Date'] = pd.to_datetime(df['Date'])
        xG['Date'] = pd.to_datetime(xG['Date'])
        df = df.merge(xG[['Date', 'Team', 'TeamxG', 'OpponentxG']], on=['Date', 'Team'])
        return df
    
    def calculate_totals(self, df, cols):
        new_cols = [f'Total{col}' for col in cols]
        for i, new_col in enumerate(new_cols):
            df[new_col] = df[f'Team{cols[i]}'] + df[f'Opponent{cols[i]}']
        
        return df, new_cols
    
    def create_stats(self, df):
        if self.xg:
            total_cols = ['Goals', 'xG', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards', 'ELO']
            total_cols1 = ['Goals', 'xG', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards']
            ind_cols = ['TeamGoals', 'OpponentGoals', 'TeamxG', 'OpponentxG', 'TeamShots', 'OpponentShots', 'TeamFouls', 'OpponentFouls','TeamShotsonTarget', 'OpponentShotsonTarget', 'TeamCorners', 'OpponentCorners', 'TeamYellowCards', 'OpponentYellowCards','TeamRedCards','OpponentRedCards']

        else:
            total_cols = ['Goals', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards', 'ELO']
            total_cols1 = ['Goals', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards']
            ind_cols = ['TeamGoals', 'OpponentGoals', 'TeamShots', 'OpponentShots', 'TeamFouls', 'OpponentFouls','TeamShotsonTarget', 'OpponentShotsonTarget', 'TeamCorners', 'OpponentCorners', 'TeamYellowCards', 'OpponentYellowCards','TeamRedCards','OpponentRedCards']
        pred_cols = []
        
        df = self.get_elo_data(df)
        if self.xg:
            df = self.add_xg(df)
        df, new_cols = self.calculate_totals(df, total_cols)
        df, new_cols = self.create_rolling(df, new_cols)
        pred_cols.extend(new_cols)
        df, new_cols = self.create_rolling(df, ind_cols)
        pred_cols.extend(new_cols)
        print(new_cols)
        df, dif_cols = self.calculate_difs(df, total_cols1, rolling=True)
        print(dif_cols)
        pred_cols.extend(dif_cols)
        return df, pred_cols
    
    def random_forest_win(self,data, predictors, train, test, call_type, target_col, model_type):  
        if model_type == 'RF':
            model = RandomForestClassifier(n_estimators=50, min_samples_split=7, random_state=1) 
        else: 
            params = {'colsample_bytree': 0.5, 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 5, 'reg_lambda': 0, 'subsample': 0.8}
            model = xgb.XGBClassifier(**params)#, objective='binary:logistic')

        #model = ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.8, min_samples_leaf=17, min_samples_split=5, n_estimators=100)
        '''
        param_grid = {
            "max_depth": [3],
            "learning_rate": [0.01],
            "gamma": [0, 0.25, 1],
            "reg_lambda": [0, 1, 10],
            "scale_pos_weight": [1, 3, 5],
            "subsample": [0.8],
            "colsample_bytree": [0.5],
        }
        '''
        param_grid = {
            "max_depth": [3],
            "learning_rate": [0.01],
            "gamma": [0, 0.25, 1],
            "reg_lambda": [0, 1, 10],
            "subsample": [0.8],
            "colsample_bytree": [0.5],
        }

        params = {'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.01, 'max_depth': 3, 'reg_lambda': 10, 'scale_pos_weight': 1, 'subsample': 0.8}
        params = {'colsample_bytree': 0.5, 'gamma': 0, 'learning_rate': 0.01, 'max_depth': 5, 'reg_lambda': 0, 'subsample': 0.8}
        #params = {'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.01, 'max_depth': 3, 'reg_lambda': 10, 'subsample': 0.8}
        
        #model = xgb.XGBClassifier(**params)#, objective='binary:logistic')
        #model = xgb.XGBRegressor()
        #grid_cv = GridSearchCV(model, param_grid, n_jobs=-1, cv=3, scoring='roc_auc')
        #_ = grid_cv.fit(train[predictors], train[target_col])
        #print(grid_cv.best_score_, grid_cv.best_params_)

        #model.fit(train[predictors], train['Target'])
        model.fit(train[predictors], train[target_col])
        preds = model.predict(test[predictors])
        #return preds,1,1
        #if call_type == 'predict':
        #    return preds, model
        #combined = pd.DataFrame(dict(actual=test['Target'], prediction=preds), index = test.index)
        #print(test.groupby('GameID').filter(lambda x: x['prediction'].sum()==1))
        precision = precision_score(test[target_col], preds, average='micro')
        accuracy = accuracy_score(test[target_col], preds)
        combined = pd.DataFrame(dict(actual=test[target_col], prediction=preds), index = test.index)
        c1 = combined.merge(data[['Date', 'Team', 'Opponent', 'TeamGoals', 'OpponentGoals','GameID', 'Venue']], left_index=True, right_index=True)

        # predictions where both home and away team prediction match
        predictions = c1.groupby('GameID').filter(lambda x: x['prediction'].sum()==1)
        #print(predictions)
        table = c1[['GameID','Date','Team', 'Opponent', 'prediction', 'actual']].merge(c1[['GameID','Date','Team', 'Opponent', 'prediction']], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])

        table = table[table['prediction_x'] == table['prediction_y']]#.filter(lambda x: x['prediction_x'] != x['prediction_y'])
        precision1 = precision_score(table['actual'], table['prediction_x'], average='micro')
        accuracy1 = accuracy_score(table['actual'], table['prediction_x'])
        values_to_write = [f'League-{self.league}',
            f'Target-{target_col}', f'Precision1-{precision1}', 
            f'Accuracy1- {accuracy1}', f'Precision-{precision}', 
            f'Accuracy-{accuracy}']
        with open('models/model_stats.txt', 'w') as f:
            for line in values_to_write:
                f.write(line)
                f.write('\n')
        
        #precision, accuracy = 1,1
        #precision = cross_val_score(model, train[predictors], train['TotalGoals'], scoring='neg_mean_absolute_error')
        return table, precision1, accuracy1, model, combined

    def prep_data(self, rolling, date):
        if self.xg:
            total_cols = ['Goals', 'xG', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards', 'ELO']
            total_cols1 = ['Goals', 'xG', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards']
            ind_cols = ['TeamGoals', 'OpponentGoals', 'TeamxG', 'OpponentxG', 'TeamShots', 'OpponentShots', 'TeamFouls', 'OpponentFouls','TeamShotsonTarget', 'OpponentShotsonTarget', 'TeamCorners', 'OpponentCorners', 'TeamYellowCards', 'OpponentYellowCards','TeamRedCards','OpponentRedCards']

        else:
            total_cols = ['Goals', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards', 'ELO']
            total_cols1 = ['Goals', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards']
            ind_cols = ['TeamGoals', 'OpponentGoals', 'TeamShots', 'OpponentShots', 'TeamFouls', 'OpponentFouls','TeamShotsonTarget', 'OpponentShotsonTarget', 'TeamCorners', 'OpponentCorners', 'TeamYellowCards', 'OpponentYellowCards','TeamRedCards','OpponentRedCards']
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
        #print(schedule['Date'], sched['Date'],date)
        #print(sched.shape, 'hi')
        sched['Result'] = -1
        sched = self.rename_cols(sched)
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
        if self.xg:
            sched = sched.add_xg()
        print(sched.head(10))
        sched = self.home_away(sched)
        

        sched, _ = self.calculate_totals(sched, ['ELO'])
        sched, _ = self.calculate_difs(sched, ['ELO'], rolling=False)
        
        sched = sched.sort_values('Date')
        #sched, _ = self.create_rolling(sched, ['TotalELO'])
        #return sched
        
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
        #['Date'].unique(),'hi')
        #new['TotalGoals'] = new['TeamGoals'] + new['OpponentGoals']
        new['Date'] = pd.to_datetime(new['Date'], dayfirst=True).dt.date
        #total_cols = ['Goals', 'xG', 'Shots', 'Fouls', 'ShotsonTarget', 'Corners', 'YellowCards', 'RedCards', 'ELO']
        #new[new['Date'] > date]['ELO']
        #new = self.calculate_totals(new)
        #return new
        #new = self.create_predictors(new)
        new, new_cols = self.calculate_totals(new, total_cols)
        new, new_cols = self.create_rolling(new, new_cols)
        new, new_cols = self.create_rolling(new, ind_cols)
        new, dif_cols = self.calculate_difs(new, total_cols1)
        new = self.create_predictors(new)

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
        new.to_csv(f'cache/{self.league}/new.csv')
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

    def train(self, date, pred_type, predictors):
        #new['WinTarget'] = codes.cat.codes
        #cats = codes.cat.categories
        #print(new['WinTarget'], new['Result'])
        #return new, 'hi'
        new = pd.read_csv(f'cache/{self.league}/new.csv')
        
        #predictors = ['VenueCode', 'OpponentCode', 'ELODif', 'Day'] + dif_cols
        #predictors = ['VenueCode', 'ELODif']
        #print(new)
        if pred_type == 'WinTarget':
            new_df = new.dropna(subset=['TeamWinOdds'])
        else:
            new_df = new.dropna(subset=['O2.5'])
        #new_df['TotalGoals'] = new_df['TotalGoals'].astype('int')
        #new['TotalGoals'] = new['TotalGoals'].astype('int')
        train = new_df[new_df['Date'] < '2018-09-18']
        #return train
        #test = new_df[new_df['Date'] >= '2022-08' and new_df['Date'] < '2022-08']
        #new_df.Date
        test = new_df[(new_df['Date'] >= '2018-09-18') & (new_df['Date'] < '2023-09')]
        #pred = new.loc[(new['Date'] >= date)]
        #pred_type = 'WinTarget'
        table, prec, acc, model, combined = self.random_forest_win(df_rolling, predictors, train, test, 'predict', pred_type, 'RF')
        print(prec, acc)
        pickle.dump(model, open(self.filename, 'wb'))
        #print(combined)
        #preds = model.predict(pred[predictors])
        #c1 = preds.merge(test[['Date', 'Team', 'Opponent', 'TotalGoals', 'GameID']], left_index=True, right_index=True)   
        return table

    def predict(self, predictors):
        new = pd.read_csv(f'cache/{self.league}/new.csv')
        
        model = pickle.load(open(self.filename, 'rb'))
        pred = new.loc[(new['Date'] >= date)]
        #pred = pred[predictors]
        #print(dict(pred.isna().sum()), pred.shape, pred.dropna().shape, predictors)

        preds = model.predict(pred[predictors])
        return preds, pred


    #@cache_to_csv(f'Predictions/Daily/{date}_predictions.csv', refresh_time=r_time)
    def daily_predictions(self):
        today = d.today()
        targets = ['WinLoseTarget', 'OU1.5Target', 'OU2.5Target', 
        'OU3.5Target', 'WinTarget', 'DrawTarget', 'LossTarget']

        leagues = ['PremierLeague', 'LaLiga', 'SerieA']
        dfs = [pd.DataFrame() for target in targets]
        #for target in targets:
        #    df_
        #df = pd.DataFrame()
        for league in leagues:
            for i,target in enumerate(targets):
                path = f'Predictions/{league}/{league}_{target}.csv'
                print(path)
                predictions = pd.read_csv(path)
                predictions['League'] = league
                print(predictions.shape)
                dfs[i] = pd.concat([dfs[i], predictions])
        delta = today + timedelta(weeks=1)
        for i,df in enumerate(dfs):
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            df = df[df['Date'] <= delta].sort_values(['Date', 'League'])
        
            df.to_csv(f'Predictions/Daily/{today}_{targets[i]}_predictions.csv')
    
    



#table[table[('preds_x' == 'preds_y')]]

if __name__ == "__main__":
    '''
    fbref = sd.WhoScored(leagues='ITA-Serie A')
    sched23 = fbref.read_missing_players()
    '''
    league = 'PremierLeague'
    #league = 'SerieA'
    #league = 'LaLiga'
    train = True
    xg = False
    func = Functions(league, xg=xg)

    #func.get_league_data()

    print('getting league')
    prem_df = func.get_league_data()
    print('getting schedule')
    sched23 = func.get_current_season()
    new_df2023 = pd.concat([prem_df, sched23])


    print('preprocess')
    df1 = func.preprocess(new_df2023)
    #df1 = func.add_xg(df1)
    # creates totals columns and adds rolling totals and rolling team stats
    df, rolling_cols = func.create_stats(df1)
    df_rolling = func.create_predictors(df)
    df_rolling.to_csv(f'cache/{league}/rolling.csv')
    print('predicting')
    #predictors = ['VenueCode', 'OpponentCode', 'ELODif']
    #ew_df = df_rolling.dropna(subset=['O2.5'])
    #train = new_df[new_df['Date'] < '2023-08']
    #test = new_df[new_df['Date'] >= '2023-08']
    #combined, precision, accuracy = func.make_predictions(df_rolling, predictors+new_cols, train, test)
    #precision, accuracy, combined
    date = str(d.today())#'2023-09-26'

    dif_cols = func.prep_data(df_rolling, date)#['30/09/23', '01/10/23', '02/10/23'])
    #predictors = ['VenueCode', 'OpponentCode', 'ELODif', 'Day'] + dif_cols
    if xg:
        predictors = ['TotalShotsRolling', 'TeamGoalsRolling', 'TeamFoulsRolling', 'GoalsRolling_Dif', 'ShotsonTargetRolling_Dif', 'OpponentShotsonTargetRolling', 'TotalELORolling', 'xGRolling_Dif', 'ShotsRolling_Dif', 'YellowCardsRolling_Dif', 'TeamRedCardsRolling', 'TotalRedCardsRolling', 'OpponentYellowCardsRolling', 'ELODif', 'TotalFoulsRolling', 'TeamShotsonTargetRolling', 'OpponentCornersRolling', 'TotalCornersRolling', 'FoulsRolling_Dif', 'OpponentGoalsRolling', 'RedCardsRolling_Dif', 'TotalGoalsRolling', 'TeamShotsRolling', 'CornersRolling_Dif', 'TeamCornersRolling', 'OpponentCode', 'TotalYellowCardsRolling', 'VenueCode', 'OpponentRedCardsRolling', 'OpponentShotsRolling', 'TeamYellowCardsRolling', 'TotalShotsonTargetRolling', 'DayCode', 'OpponentFoulsRolling']
    else:
        predictors = ['TotalShotsRolling', 'TeamGoalsRolling', 'TeamFoulsRolling', 'GoalsRolling_Dif', 'ShotsonTargetRolling_Dif', 'OpponentShotsonTargetRolling', 'TotalELORolling', 'ShotsRolling_Dif', 'YellowCardsRolling_Dif', 'TeamRedCardsRolling', 'TotalRedCardsRolling', 'OpponentYellowCardsRolling', 'ELODif', 'TotalFoulsRolling', 'TeamShotsonTargetRolling', 'OpponentCornersRolling', 'TotalCornersRolling', 'FoulsRolling_Dif', 'OpponentGoalsRolling', 'RedCardsRolling_Dif', 'TotalGoalsRolling', 'TeamShotsRolling', 'CornersRolling_Dif', 'TeamCornersRolling', 'OpponentCode', 'TotalYellowCardsRolling', 'VenueCode', 'OpponentRedCardsRolling', 'OpponentShotsRolling', 'TeamYellowCardsRolling', 'TotalShotsonTargetRolling', 'DayCode', 'OpponentFoulsRolling']

    #targets = ['WinLoseTarget', 'OU1.5Target', 'OU2.5Target', 
    #'OU3.5Target', 'WinTarget', 'DrawTarget', 'LossTarget']
    targets = ['OUCornersTarget']
    #targets = ['WinLoseTarget', 'Target', 'WinTarget', 'DrawTarget', 'LossTarget']

    for target in targets:
        #print(target)
        if train:
            combined = func.train(date, target, predictors)
            preds, c = func.predict(predictors)
        else:
            preds, c = func.predict(predictors)
        c['preds'] = preds
        table = c.sort_index()[['GameID','Date','Team', 'Opponent', 'preds']].merge(c[['GameID','Date','Team', 'Opponent', 'preds']], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])
        table.to_csv(f'Predictions/{league}/{league}_{target}.csv')
    
    #func.daily_predictions()
    '''
    
    preds_o_u, combined0, c0 = func.predict(date, 'TotalGoals', dif_cols)
    preds_wl, combined1, c1 = func.predict(date, 'WinLoseTarget', dif_cols)
    preds_ou, combined2, c2 = func.predict(date, 'OUTarget2.5', dif_cols)
    preds_w, combined3, c3 = func.predict(date, 'WinTarget', dif_cols)
    preds_d, combined4, c4 = func.predict(date, 'DrawTarget', dif_cols)
    preds_l, combined5, c5 = func.predict(date, 'LossTarget', dif_cols)

    c0['preds'] = preds_o_u
    table_o_u = c0.sort_index()[['GameID','Date','Team', 'Opponent', 'preds']].merge(c0[['GameID','Date','Team', 'Opponent', 'preds']], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])
    c1['preds'] = preds_wl
    table_wl = c1.sort_index()[['GameID','Date','Team', 'Opponent', 'preds']].merge(c1[['GameID','Date','Team', 'Opponent', 'preds']], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])
    c2['preds'] = preds_ou
    table_ou = c2.sort_index()[['GameID','Date','Team', 'Opponent', 'preds']].merge(c2[['GameID','Date','Team', 'Opponent', 'preds']], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])
    c3['preds'] = preds_w
    table_w = c3.sort_index()[['GameID','Date','Team', 'Opponent', 'preds']].merge(c3[['GameID','Date','Team', 'Opponent', 'preds']], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])
    c5['preds'] = preds_l
    table_l = c5.sort_index()[['GameID','Date','Team', 'Opponent', 'preds']].merge(c5[['GameID','Date','Team', 'Opponent', 'preds']], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])
    c4['preds'] = preds_d
    table_d = c4.sort_index()[['GameID','Date','Team', 'Opponent', 'preds']].merge(c4[['GameID','Date','Team', 'Opponent', 'preds']], left_on=['Date', 'Team'], right_on=['Date', 'Opponent']).drop_duplicates(['GameID_x'])
    table_wl.to_csv(league + '/w_l.csv')
    
    table_ou.to_csv(league + '/o_u.csv')
    table_o_u.to_csv(league + '/_o_u.csv')
    table_w.to_csv(league + '/w.csv')
    table_d.to_csv(league + '/d.csv')
    table_l.to_csv(league + '/l.csv')
    '''


    
