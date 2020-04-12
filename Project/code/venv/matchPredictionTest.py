import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

############################################### Data Loading ###########################################################


df_list_PL = [pd.read_csv('../data/PREMIER_LEAGUE/PL_'+str(year)+'.csv') for year in np.arange(2013,2019)]
df_list_E1 = [pd.read_csv('../data/CHAMPIONSHIP/E1_'+str(year)+'.csv') for year in np.arange(2013,2019)]
df_list_E2 = [pd.read_csv('../data/LEAGUE1/E2_'+str(year)+'.csv') for year in np.arange(2013,2019)]


df_list = df_list_PL + df_list_E1 + df_list_E2


data_PL = pd.concat(df_list_PL, ignore_index=True)
data_E1 = pd.concat(df_list_E1, ignore_index=True)
data_E2 = pd.concat(df_list_E2, ignore_index=True)

data = pd.concat(df_list, ignore_index=True)



data.isnull().sum().plot()

# Feature with more than 10% of NaN
NaN_feature = data.isnull().sum()[data.isnull().sum()>0.10*len(data)].index

# Drop them
data.drop(columns = NaN_feature, inplace=True)
# Drop row with nan
data.dropna(axis=0,how='any',inplace=True)

data.isnull().sum().sum()

data.dtypes
data.describe()

data.columns
odds_columns = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD',
                'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA', 'PSCH', 'PSCD', 'PSCA']


data[['PSH', 'PSD','PSA','PSCH', 'PSCD', 'PSCA']]
