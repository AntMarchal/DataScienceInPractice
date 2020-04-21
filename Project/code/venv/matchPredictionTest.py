import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns

############################################### Data Loading ###########################################################


df_list_PL = [pd.read_csv('../data/PREMIER_LEAGUE/PL_'+str(year)+'.csv') for year in np.arange(2013,2019)]
df_list_E1 = [pd.read_csv('../data/CHAMPIONSHIP/E1_'+str(year)+'.csv') for year in np.arange(2013,2019)]
df_list_E2 = [pd.read_csv('../data/LEAGUE1/E2_'+str(year)+'.csv') for year in np.arange(2013,2019)]


df_list = df_list_PL + df_list_E1 + df_list_E2


data = pd.concat(df_list, ignore_index=True)

# Convert date to datetime
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(by='Date',inplace=True)
data.reset_index(drop=True,inplace=True)



############################################### Some Cleaning ##########################################################
# Feature with more than 10% of NaN
NaN_feature = data.isnull().sum()[data.isnull().sum()>0.10*len(data)].index

# Drop them
data.drop(columns = NaN_feature, inplace=True)
# Drop row with nan
data.dropna(axis=0,how='any',inplace=True)

data.isnull().sum().sum()
################################################# EDA ##################################################################


plt.pie(y.value_counts(),autopct='%1.1f%%')
plt.legend(['Home','Away','Draw'])
plt.title('Class repartition')
plt.show()


############################################### Feature engineering ####################################################

odds_columns = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD',
                'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'PSCH', 'PSCD', 'PSCA']
goal_odds_columns = ['BbOU', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5','BbAv<2.5']
Asian_odds_columns = ['BbAH', 'BbAHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA']
Stat_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS','HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']


# Take the inverse of odds \equiv implied prob
data[odds_columns] = data[odds_columns].apply(lambda x:1./x).replace(np.inf,np.nan)
# remove infinite
data.dropna(axis=0, how='any',inplace=True)
# Reset again index in order to avoid hole
data.reset_index(drop=True,inplace=True)

# add implied proba min
H_odds_columns = [column for column in odds_columns if column[-1]=='H']
D_odds_columns = [column for column in odds_columns if column[-1]=='D']
A_odds_columns = [column for column in odds_columns if column[-1]=='A']


data['H_proba_min'] = data[H_odds_columns].min(axis=1)
data['D_proba_min'] = data[D_odds_columns].min(axis=1)
data['A_proba_min'] = data[A_odds_columns].min(axis=1)

data.drop(['Bb1X2', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD','BbMxA', 'BbAvA'], axis=1, inplace=True)



#Momentum in stat and victory

data_mom = data[['Date','HomeTeam','AwayTeam']+Stat_columns]

data_victory = data[['Date','HomeTeam','AwayTeam','FTR']]

Home_columns = ['FTHG', 'HTHG', 'HS', 'HST', 'HF', 'HC', 'HY', 'HR']
Away_columns = ['FTAG', 'HTAG', 'AS', 'AST', 'AF', 'AC', 'AY', 'AR']

def Momentum(data_past,team):
    new_columns = ['Date', 'FTG', 'HTG', 'S', 'ST', 'F', 'C', 'Y', 'R']
    Home_columns = ['FTHG', 'HTHG', 'HS', 'HST', 'HF', 'HC', 'HY', 'HR']
    Away_columns = ['FTAG', 'HTAG', 'AS', 'AST', 'AF', 'AC', 'AY', 'AR']
    data_past_H = data_past[data_past['HomeTeam'] == team][['Date']+Home_columns]
    data_past_H.columns = new_columns
    data_past_A = data_past[data_past['AwayTeam'] == team][['Date']+Away_columns]
    data_past_A.columns = new_columns
    mom_df = pd.merge_ordered(data_past_H,data_past_A).ewm(span=10).mean()
    if mom_df.empty:
        mom_df.loc[0,mom_df.columns] = np.nan
    match_nbr = data_past_H.shape[0] + data_past_A.shape[0]
    return mom_df.iloc[-1,:], match_nbr


def VictoryHistory(data_victory_past,team1,team2):
    victories = list()
    draws = list()

    victories += (data_victory_past[(data_victory_past['HomeTeam'] == team1)
                                        & (data_victory_past['AwayTeam'] == team2)]['FTR']=='H').tolist()
    draws += ((data_victory_past[(data_victory_past['HomeTeam'] == team1)
                                        & (data_victory_past['AwayTeam'] == team2)]['FTR']=='D').tolist())

    victories += ((data_victory_past[(data_victory_past['HomeTeam'] == team2)
                                        & (data_victory_past['AwayTeam'] == team1)]['FTR']=='A').tolist())
    draws += ((data_victory_past[(data_victory_past['HomeTeam'] == team2)
                                        & (data_victory_past['AwayTeam'] == team1)]['FTR']=='D').tolist())
    confrontation_nbr = len(victories)

    if confrontation_nbr == 0:
        victory_mom = 0
        draw_mom =0
    else:
        victory_mom = pd.Series(victories).ewm(5).mean().iloc[-1]
        draw_mom = pd.Series(draws).ewm(5).mean().iloc[-1]
    return victory_mom,draw_mom,confrontation_nbr

data[Stat_columns] = np.nan
data['match_nbr_home'] = np.nan
data['match_nbr_away'] = np.nan
data['H_vict_mom'] = np.nan
data['D_mom'] = np.nan
data['confrontation_nbr'] = np.nan

for i in range(1,data.shape[0]):
    data_mom_past = data_mom.iloc[:i]
    momH, match_nbr_home = Momentum(data_mom_past,data['HomeTeam'][i])
    data.loc[i,Home_columns] = momH.values
    data.loc[i, 'match_nbr_home'] = match_nbr_home

    momA, match_nbr_Away = Momentum(data_mom_past, data['AwayTeam'][i])
    data.loc[i, Away_columns] = momA.values
    data.loc[i, 'match_nbr_away'] = match_nbr_Away

    data_victory_past = data_victory.iloc[:i]
    H_vict_mom,D_mom,confrontation_nbr = VictoryHistory(data_victory_past,data['HomeTeam'][i],data['AwayTeam'][i])
    data.loc[i, 'H_vict_mom'] = H_vict_mom
    data.loc[i, 'D_mom'] = D_mom
    data.loc[i, 'confrontation_nbr'] = confrontation_nbr

# Drop Nan that comes from lack of data in momentum
data.dropna(axis=0, how='any',inplace=True)


y = data['FTR']
X = data.drop(columns=['Div','Date','HomeTeam','AwayTeam','FTR','HTR'],axis=1)

################################################# Scaling ##############################################################
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
################################################### Feature Importance #################################################
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=50)
model.fit(X, y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values()
feat_importances.plot(kind='barh',legend=False,figsize=(12,9))
plt.show()
################################################### PCA ################################################################
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X)
X = pca.transform(X)
Cum_explained_Var = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(10, 5))
ax = sns.scatterplot(data=Cum_explained_Var)
ax.set(xlabel='Dimensions', ylabel='Explained Variance Ratio')
plt.title("PCA-transformed cumulated variance explained")
plt.show()
# from the 30th component, we don't really add information
X = X[:,:30]
X.shape
############################################### Model Construction #####################################################

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, plot_confusion_matrix, classification_report, accuracy_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.25)

classifier_names = ['clf_KNN','clf_logistic', 'clf_SVC', 'clf_RF', 'clf_GNB']

clf_KNN = KNeighborsClassifier()
clf_logistic = LogisticRegression(multi_class = "ovr", class_weight = 'balanced')
clf_SVC = SVC(class_weight = 'balanced',probability=True)
clf_RF = RandomForestClassifier(class_weight = 'balanced')
clf_GNB = GaussianNB()

clf_list = [clf_KNN, clf_logistic, clf_SVC, clf_RF, clf_GNB]

param_KNN = {'n_neighbors': [9, 10, 11]}
param_logistic = {'C': np.logspace(-3,1,5)}
param_SVC = {'C': [6,8,12,20]}
param_RF = {'n_estimators':[10,50],'max_depth':[10,50], 'max_features':['auto', 'log2'],'min_samples_split':[2,5]}
param_GNB = {}

param_list = [param_KNN, param_logistic, param_SVC, param_RF, param_GNB]

CrossVal_dict = {clf_name:{'Classifier':clf,'Hyperparameter':param} for clf_name,clf,param
                 in zip(classifier_names,clf_list,param_list)}

############################################## Model Selection #########################################################

def custom_score_fct(y_true,y_pred):
    '''
    Rank Probability score
    :param y_true: (n_sample,) vector of true class A, D or H
    :param y_pred: (n_sample,n_class) matrix of proba of the respective class A, D, or H
    :return: score
    '''
    n_sample = y_pred.shape[0]
    e1 = np.ones(n_sample) * (y_true == 'A')
    e2 = np.ones(n_sample) * (y_true == 'D')
    p1 = y_pred[:,0]
    p2 = y_pred[:, 1]
    def RPS(e1,e2,p1,p2):
        return 0.5 * ((p1-e1)**2 + (p1+p2-e1-e2)**2)
    return np.mean(list(map(RPS,e1,e2,p1,p2)))


RPS_score = make_scorer(custom_score_fct,greater_is_better=False,needs_proba=True)


def ModelSelection(CrossVal_dict,score_fct = 'f1_weighted'):
    Best_Classifiers_dict = {}

    for clf in CrossVal_dict.keys():

        CV = GridSearchCV(CrossVal_dict[clf]['Classifier'], scoring=score_fct,
                          param_grid=CrossVal_dict[clf]['Hyperparameter'], cv=5)
        CV = CV.fit(X_train, y_train)
        Best_Classifiers_dict[clf] = {'Classifier': CV.best_estimator_, 'Hyperparameter': CV.best_params_}

    return Best_Classifiers_dict



Best_Classifiers_dict = ModelSelection(CrossVal_dict)
Best_Classifiers_dict = ModelSelection(CrossVal_dict,score_fct=RPS_score)


############################################### Tests Results ##########################################################

def DisplayResults(Best_Classifiers_dict,X_test,y_test):
    Score_Comparison_df = pd.DataFrame(index=Best_Classifiers_dict.keys(),columns=['F1_weighted_Score','RPS'])
    for clf in Best_Classifiers_dict.keys():
        print('###############################################################')
        plot_confusion_matrix(Best_Classifiers_dict[clf]['Classifier'], X_test, y_test, cmap="YlGnBu",
                              values_format='.0f')
        plt.title('Confusion matrix ' + clf)
        plt.show()
        print('Classification report of ', clf)
        print(classification_report(y_test, Best_Classifiers_dict[clf]['Classifier'].predict(X_test)))
        print('Rank probability score: {:.2f}'.format(
            RPS_score(Best_Classifiers_dict[clf]['Classifier'], X_test, y_test)))
        print('###############################################################')
        Score_Comparison_df.loc[clf, 'F1_weighted_Score'] = f1_score(y_test, Best_Classifiers_dict[clf]['Classifier']
                                                                       .predict(X_test),average='weighted')
        Score_Comparison_df.loc[clf, 'RPS'] = RPS_score(Best_Classifiers_dict[clf]['Classifier'], X_test, y_test)

    Score_Comparison_df.abs().plot.barh(figsize=(8,6),grid=True)



DisplayResults(Best_Classifiers_dict,X_test,y_test)



# clf = 'clf_KNN'
# clf = 'clf_logistic'
# clf = 'clf_SVC'
# clf = 'clf_RF'
# clf = 'clf_GNB'




############################################### Arbitrage strategy #####################################################

data['Cs'] = 1/(data[['H_proba_min','D_proba_min','A_proba_min']].sum(axis=1))
Arbitrages = data[['Date','Cs']][data['Cs']>1]
Arbitrages.index = Arbitrages['Date']
Arbitrages.drop(columns='Date',axis=1, inplace=True)
Max_Arbitrage = Arbitrages.groupby(by='Date').max()

NbrArb_peryear = Max_Arbitrage.resample('Y').count()
NbrArb_peryear.columns = ['Nbr_arbitrage']
NbrArb_peryear.index = NbrArb_peryear.index.year
NbrArb_peryear.plot.bar(figsize=(8,6), title='Number of arbitrage opportunity per year')

(Max_Arbitrage-1).plot(figsize=(8,6), title='Arbitrage returns',legend=False)
plt.ylabel('R')

R_peryear = Max_Arbitrage.resample('Y').prod()-1
R_peryear.columns = ['R']
R_peryear.index = R_peryear.index.year
R_peryear.plot.bar(figsize=(8,6), title='Return per year')

Max_Arbitrage.cumprod().plot(figsize=(8,6), title='Cummulative performance of 1$',legend=False)
plt.ylabel('$')






