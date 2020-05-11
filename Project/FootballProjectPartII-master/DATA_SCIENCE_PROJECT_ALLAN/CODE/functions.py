import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def after(text, string):
    # Find and validate first part.
    pos_str = text.rfind(string)
    if pos_str == -1: return ""
    # Returns chars after the found string.
    adjusted_pos_str = pos_str + len(string)
    if adjusted_pos_str >= len(text): return ""
    return text[adjusted_pos_str:]

def delete_nan_column(data, max_number_of_nas):
    data = data.loc[:, (data.isnull().sum(axis=0) <= max_number_of_nas)]
    return data

def import_data(init_path, years):
    total_data=[]
    errors =[]
    for year in years:
        try:
            path = init_path+year+'.csv'
            data = pd.read_csv(path)
            total_data.append(data)
        except Exception as err:
            errors.append(err)
            bound = int(after(str(err), 'Expected')[1:3])
            path = init_path+year+'.csv'
            data = pd.read_csv(path, usecols=[i for i in range(bound)], encoding = 'unicode_escape')
            total_data.append(data)
            
    for i in range(len(total_data)):
        total_data[i] = delete_nan_column(total_data[i], 100)
                
    return total_data

def min_odd(row_data, bookmaker):
    
    if bookmaker == 'Bet365':
        odds_to_take = ['B365H', 'B365D', 'B365A']
    
    elif bookmaker == 'Bet&Win':
        odds_to_take = ['BWH', 'BWD', 'BWA']
    
    subdata = row_data[odds_to_take]
    np_subdata = subdata.to_numpy()
    array_subdata = np.reshape(np_subdata, (-1, 1))
    idx_min = np.argmin(array_subdata)
    if idx_min == 0:
        odd_min = 'H'
    elif idx_min == 1:
        odd_min = 'D'
    else:
        odd_min = 'A'
        
    return odd_min

def bookmaker_accuracy(subdata, bookmaker):
    counts = []
    for i in range(len(subdata)):
        if min_odd(subdata.iloc[i], bookmaker) == subdata['FTR'].iloc[i]:
        #subdata['FTR'].iloc[i] == 'H' and odds_to_prob(subdata['B365H'].iloc[i]) > 0.5 or subdata['FTR'].iloc[i] == 'D' and odds_to_prob(subdata['B365D'].iloc[i])>0.5 or subdata['FTR'].iloc[i] == 'A' and odds_to_prob(subdata['B365A'].iloc[i]) > 0.5:
            count = 1
        else:
            count = 0
        counts.append(count)
    accuracy = np.sum(counts)/len(subdata)
    return accuracy

def show_odds(data, bookmaker):
    if bookmaker == 'Bet365':
        odds = ['B365H', 'B365D', 'B365A']
    elif bookmaker == 'Bet&Win':
        odds = ['BWH', 'BWD', 'BWA']
    return data[odds]

def odds_to_prob(odd):
    return 1/odd

def choose_bet(data, date, bookmaker, min_odd):
    chosen_data = data[data['Date']==date]
    if bookmaker == 'Bet365':
        home = 'B365H'
        away = 'B365A'
    elif bookmaker == 'Bet&Win':
        home = 'BWH'
        away = 'BWA'
    opt_bets = []
    for i in range(len(chosen_data)):
        home_odd = chosen_data[home].iloc[i]
        away_odd = chosen_data[away].iloc[i]
        if  home_odd <= min_odd:
            opt_odd = home
            bet = chosen_data[['Date','HomeTeam', 'AwayTeam', 'FTR', opt_odd]].iloc[i]
            opt_bets.append(bet)
        elif away_odd <= min_odd:
            opt_odd = away
            bet = chosen_data[['Date','HomeTeam', 'AwayTeam', 'FTR', opt_odd]].iloc[i]
            opt_bets.append(bet)
    return opt_bets

def count_result(data):
    n_games = 380
    HomeWins = data['FTR'].str.count('H')
    n_HomeWins = np.sum(HomeWins)
    AwayWins = data['FTR'].str.count('A')
    n_AwayWins = np.sum(AwayWins)
    Draws = data['FTR'].str.count('D')
    n_Draws = np.sum(Draws)
    prob_HomeWins = n_HomeWins/n_games
    prob_AwayWins = n_AwayWins/n_games
    prob_Draw = n_Draws/n_games
    print('ratio of Home Wins: ' + str(round(prob_HomeWins*100)), '%, ratio of Away Wins: ' + str(round(prob_AwayWins*100)), '%, ratio of Draws: ' + str(round(prob_Draw*100)),'%')
    return prob_HomeWins, prob_AwayWins, prob_Draw

def bar_chart(data, time_horizon):
    fig= plt.figure(figsize=(12,7))
    width = 0.4
    plt.bar(time_horizon, data, width)
    
def find_games(data, HomeTeam, AwayTeam):
    return data.loc[(data['HomeTeam'] == HomeTeam) & (data['AwayTeam'] == AwayTeam)]
    

