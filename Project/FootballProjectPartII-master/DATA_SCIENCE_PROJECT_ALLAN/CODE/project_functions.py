import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def after(text, string):
    # Find and validate first part.
    pos_str = text.rfind(string)
    if pos_str == -1: return ""
    # Returns chars after the found string.
    adjusted_pos_str = pos_str + len(string)
    if adjusted_pos_str >= len(text): return ""
    return text[adjusted_pos_str:]

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

def delete_nan_column(data, max_number_of_nas):
    data = data.loc[:, (data.isnull().sum(axis=0) <= max_number_of_nas)]
    return data

def merge_list(data):
    df = pd.merge(data[0], data[1], 'outer')
    for i in range(2, len(data)):
        df = pd.merge(df, data[i], 'outer')
    return df

def choose_team(data, team, date):
    """
    Output: returns all the data of a certain team available at time t-1 to predict a game a time t.
    """
    df = data.copy()
    return df.loc[((df['date'] < date) & (df['HomeTeam'] == team)) | (df['date'] < date) & (df['AwayTeam'] == team)] 

def choose_hometeam(data, team, date):
    df = data.copy()
    return df.loc[((df['date'] < date) & (df['HomeTeam'] == team))]

def choose_awayteam(data, team, date):
    df = data.copy()
    return df.loc[((df['date'] < date) & (df['AwayTeam'] == team))]


def last_results(team, game): 
    
    w_bonus = 1 #+1 for a Win
    d_bonus = 0 #0 for a Draw
    l_malus = - w_bonus #-1 for a Loss
    
    if game['HomeTeam'] == team:
        if game['FTR'] == 1:
            global_perf = w_bonus
        elif game['FTR'] == 2:
            global_perf = l_malus
        else:
            global_perf = d_bonus
        
    if game['AwayTeam'] == team:
        if game['FTR'] == 2:
            global_perf = w_bonus
        elif game['FTR'] == 1:
            global_perf = l_malus
        else:
            global_perf = d_bonus

    return global_perf

def EWMA(data, team, date, feature, gamma):
    
    subdata = choose_team(data, team, date) #all data available at t-1 to use to predict outcome at date t
        
    perfs=[]
    for i in range(len(subdata)):
        previous_game = subdata.iloc[i] #compute performance for all games before game at date t
        if feature == 'FTRH' or feature == 'FTRA':
            perfs.append(last_results(team, previous_game))
        else:
            perfs.append(previous_game[feature]) #stores performances of all games that happened until t-1
    
    #gamma = 0.01
    n = []
    d = []
    perf = [i for i in reversed(perfs)] #Now, perform an EWMA. To do so, we need to reverse the list, because
    # we go from the most recent observation (i.e. game) to the earliest one. 
    
    for i in range(len(perf)):
        #Apply EWMA formula
        coef = (1 - gamma)**i
        nominator = perf[i] * coef
        n.append(nominator)
        denominator = coef
        d.append(denominator)
    momentum = sum(n) / sum(d)
    
    return momentum

def clean_dataset(data):
    new_data = data.copy()
    from datetime import datetime
    
    dates = []
    for i in range(len(new_data)):
        date = new_data['Date'].iloc[i]
        try:   
            date = datetime.strptime(date, '%d/%m/%y')
        except:
            date = datetime.strptime(date, '%d/%m/%Y')   
        dates.append(date)
    dates = pd.DataFrame(dates, columns=['Date'])
    new_data['date'] = dates
    
    new_data = new_data[['date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS',
                         'AS', 'HST', 'AST', 'HC', 'AC','HF', 'AF', 'B365H', 'B365D', 
                         'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 
                         'LBA', 'PSH', 'PSD','PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 
                         'VCA', 'PSCH', 'PSCD', 'PSCA', 'BbAv<2.5', 'BbAv>2.5']]
    
    new_data.dropna(inplace=True)
    new_data.reset_index(inplace=True)
    new_data = new_data.drop(['index'], axis=1)
    
    return new_data

def get_labels(data, label):
    Y = data[label]
    Y = np.array(Y)
    Y=Y.astype('int')
    
    return Y

def create_momentum_features(data, gamma, interval):
    
    k = interval[0]
    l = interval[1]
    
    data_bis = data.copy()
    
    game_features = ['FTRH', 'FTRA', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF']
    
    for feature in game_features:
        data_bis[feature + ' Momentum'] = np.nan
        
    home_features = ['FTRH', 'FTHG', 'HS', 'HC', 'HST', 'HF']
    away_features = ['FTRA', 'FTAG', 'AS', 'AC', 'AST', 'AF']

    
    for i in range(k, l):
        try:
            date = data_bis['date'].iloc[i]
            hometeam = data_bis['HomeTeam'].iloc[i]
            awayteam = data_bis['AwayTeam'].iloc[i]
            for feature in home_features:
                data_bis[feature + ' Momentum'].iloc[i] = EWMA(data_bis, hometeam, date, feature, gamma)
            for feature in away_features:
                data_bis[feature + ' Momentum'].iloc[i] = EWMA(data_bis, awayteam, date, feature, gamma)

        except:
            print(i)
           
    new_data  = data_bis.dropna()
    
    return new_data

def get_features(data, features, normalize=True):
    from sklearn.preprocessing import normalize
    X = np.array(data[features])
    if normalize:
        X = normalize(X)
    else:
        X = X
    
    return X

def convert_labels(data):
    
    df = data.copy()
    
    df.loc[df.FTR == "H", "FTR"] = 1 #Replace nominal target variables by numbers
    df.loc[df.FTR == "D", "FTR"] = 0
    df.loc[df.FTR == "A", "FTR"] = 2
    
    return df

def regression(dataset, algo, X, y1, y2, Kfold, train_set):
    import random
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    
    test_accuracies=[]
    models_home=[]
    models_away=[]
    
    for n in random.sample(range(1,1000), Kfold):
        X_train1, X_test1, y_home_train, y_home_test = train_test_split(X, y1, train_size=train_set,random_state=n)
        X_train2, X_test2, y_away_train, y_away_test = train_test_split(X, y2, train_size=train_set,random_state=n)

        model_home = algo().fit(X_train1, y_home_train)
        models_home.append(model_home)
        model_away = algo().fit(X_train2, y_away_train)
        models_away.append(model_away)
        
    models = []
    for homeModel in models_home:
        for awayModel in models_away:
            
            y_home_hat = homeModel.predict(X).reshape((-1,1))
            y_home_hat = np.round(y_home_hat)
    
            y_away_hat = awayModel.predict(X).reshape((-1,1))
            y_away_hat = np.round(y_away_hat)
        
            home_coef = homeModel.coef_
            home_intercept = homeModel.intercept_
            w_home = np.insert(np.array(home_coef).reshape(-1,1), 0, home_intercept, axis=0)
        
            away_coef = awayModel.coef_
            away_intercept = awayModel.intercept_
            w_away = np.insert(np.array(away_coef).reshape(-1,1), 0, away_intercept, axis=0)
    
    
    
    
            predictions = []
            for i in range(len(y_home_hat)):
                if y_home_hat[i] > y_away_hat[i]:
                    predictions.append(1)
                elif y_home_hat[i] < y_away_hat[i]:
                    predictions.append(2)
                else:
                    predictions.append(0)
    
            predictions = np.array(predictions).reshape(-1,1)
            true_result = np.array(dataset['FTR']).astype('int').reshape(-1,1)
    
            accuracy = len(predictions[predictions==true_result])/len(predictions)
            models.extend(([accuracy], [w_home], [w_away]))
            
    models = np.array(models).reshape(-1, 3)
    idx = np.argmax(models[:,0])
    best_model = models[idx,:]
    
    return best_model

def optimal_regressor(algo, X, y, kfold, metrics):
    
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    kf = KFold(n_splits=kfold, shuffle=False)
    kf.split(X)
    
    models = []
    
    
    for train_index, test_index in kf.split(X):
        # Split train-test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if algo == LassoCV or algo == RidgeCV:
            model = algo(cv=kfold).fit(X_train, y_train.ravel())
        elif algo == LinearRegression:
            model = algo().fit(X_train, y_train.ravel())
        else:
            print('This algorithm is not available')
        
        y_pred = model.predict(X_test)
        
        if metrics == 'MSE':
            metric = mean_squared_error(y_test, y_pred, squared=True)
        elif metrics == 'RMSE':
            metric = mean_squared_error(y_test,y_pred, squared=False)
        elif metrics == 'MAE':
            metric = mean_absolute_error(y_test, y_pred)
        else:
            print('This metric is not available')
            
        error = metric
        if algo == LassoCV:
            algo_name = 'Lasso'
        elif algo == RidgeCV:
            algo_name = 'Ridge'
        elif algo == LinearRegression:
            algo_name = 'LinearRegression'
        else:
            print('This algorithm is not available')
            
        models.extend(([model], [algo_name], [error]))
        
    models = np.array(models).reshape(-1,3)
    idx = np.argmin(models[:, 1])
    opt_model = models[idx,:]
        
    return opt_model
    