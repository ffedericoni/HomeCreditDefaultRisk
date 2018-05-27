
# coding: utf-8

# # Keras Classifier for Home Credit Default Risk

import os
import gc
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
print("Tensorflow version = ", tf.__version__)


# In[14]:

input_dir = '../input'
print('Input files:\n{}'.format(os.listdir(input_dir)))
print('Loading data sets...')

sample_size = 100000
MULT=10
app_train_df = pd.read_csv(os.path.join(input_dir, 'application_train.csv'), nrows=sample_size)
app_test_df = pd.read_csv(os.path.join(input_dir, 'application_test.csv'), nrows=sample_size)
bureau_df = pd.read_csv(os.path.join(input_dir, 'bureau.csv'), nrows=sample_size*MULT)
bureau_balance_df = pd.read_csv(os.path.join(input_dir, 'bureau_balance.csv'), nrows=sample_size*MULT)
credit_card_df = pd.read_csv(os.path.join(input_dir, 'credit_card_balance.csv'), nrows=sample_size*MULT)
pos_cash_df = pd.read_csv(os.path.join(input_dir, 'POS_CASH_balance.csv'), nrows=sample_size*MULT)
prev_app_df = pd.read_csv(os.path.join(input_dir, 'previous_application.csv'), nrows=sample_size*MULT)
install_df = pd.read_csv(os.path.join(input_dir, 'installments_payments.csv'), nrows=sample_size*MULT)

gc.collect()
DEBUG = True
# In[15]:

def printlog(string, mode="LOG"):
    """ Function to print a log string with a prefix """
    if DEBUG:
        if mode == "LOG":
            print("#### ", string)
        elif mode == "INFOBOX":
            print("\n[#########################################################################\n", 
                  string, 
                  "\n]#########################################################################\n")
        elif mode == "SILENT":
            return
    return

def bureau_bal_transform(df):
    printlog("Transforming bureau balance...")
    df = pd.concat([df, pd.get_dummies(df.STATUS, prefix='status')], axis=1).drop('STATUS', axis=1)
    buro_counts = df[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
    df['buro_count'] = df['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])
    avg_buro_bal = df.groupby('SK_ID_BUREAU').mean()
    avg_buro_bal.columns = ['avg_bb_' + f_ for f_ in avg_buro_bal.columns]
    return avg_buro_bal

bureau_balance_df = bureau_bal_transform(bureau_balance_df)
gc.collect()
printlog(bureau_balance_df.head(), "INFOBOX")

#%%
def bureau_transform(df):
    printlog("Transforming bureau ...")
    df = pd.concat([df, pd.get_dummies(df.CREDIT_ACTIVE, prefix='ca_')], axis=1).drop('CREDIT_ACTIVE', axis=1)
    df = pd.concat([df, pd.get_dummies(df.CREDIT_CURRENCY, prefix='cu_')], axis=1).drop('CREDIT_CURRENCY', axis=1)
    df = pd.concat([df, pd.get_dummies(df.CREDIT_TYPE, prefix='ty_')], axis=1).drop('CREDIT_TYPE', axis=1)
    return df

bureau_df = bureau_transform(bureau_df)
gc.collect()
printlog(bureau_df.head(), "INFOBOX")

# In[30]:
printlog('Merge bureaus...')
buro_full = bureau_df.merge(right=bureau_balance_df.reset_index(), how='left', on='SK_ID_BUREAU', 
                            suffixes=('', '_bur_bal'))

printlog('Counting buro per SK_ID_CURR')
nb_bureau_per_curr = buro_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
buro_full['SK_ID_BUREAU_cnt'] = buro_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])

printlog('Averaging bureau')
avg_buro = buro_full.groupby('SK_ID_CURR').mean()
printlog(avg_buro.head(), "INFOBOX")

for col in avg_buro.columns:
    avg_buro[col] = avg_buro[col].astype(np.float32)
    
del buro_full, bureau_df, bureau_balance_df
gc.collect()
# In[ ]:
def prevapp_transform(df):
    printlog('Process previous applications...')
    prev_cat_features = [f_ for f_ in df.columns if df[f_].dtype == 'object']
    prev_dum = pd.DataFrame()
    for f_ in prev_cat_features:
        prev_dum = pd.concat([prev_dum, pd.get_dummies(df[f_], prefix=f_).astype(np.uint8)], axis=1)
    df = pd.concat([df, prev_dum], axis=1)
    nb_prev_per_curr = df[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    df['SK_ID_PREV_cnt'] = df['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV'])
    df = df.groupby('SK_ID_CURR').mean()
    return df

prev_app_df = prevapp_transform(prev_app_df)
gc.collect()
printlog(prev_app_df.head(), "INFOBOX")
#%%
def poscash_transform(df):
    printlog('Process POS_CASH...')
    df = pd.concat([df, pd.get_dummies(df['NAME_CONTRACT_STATUS'])], axis=1)
    nb_prevs = df[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    df['SK_ID_PREV_cnt'] = df['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    df = df.groupby('SK_ID_CURR').mean()
    return df

pos_cash_df = poscash_transform(pos_cash_df)
gc.collect()
printlog(pos_cash_df.head(), "INFOBOX")

#%%
def feature_engineering(app_data, bureau_df, bureau_balance_df, credit_card_df,
                        pos_cash_df, prev_app_df, install_df):
    """ Process the dataframes into a single one containing all the transformed features """
    
    return merged_df

