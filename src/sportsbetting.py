import soccerdata as sd

fifa = sd.SoFIFA(leagues='ITA-Serie A', versions='all')
#fifa.read_teams()
#print(fifa.read_versions(max_age=1))

df = fifa.read_team_ratings()
df.to_csv('fifa.csv')
print(df)