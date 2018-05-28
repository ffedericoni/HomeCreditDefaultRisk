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

sample_size = 150000
MULT=10
bureau_df = pd.read_csv(os.path.join(input_dir, 'bureau.csv'), nrows=sample_size*MULT)
bureau_balance_df = pd.read_csv(os.path.join(input_dir, 'bureau_balance.csv'), nrows=sample_size*MULT)

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

def to_float32(df):
    for col in df.columns:
        if df[col].dtypes == np.float64:
            df[col] = df[col].astype(np.float32)
    return df
 
def bureau_bal_transform(df):
    printlog("Transforming bureau balance...")
    df = pd.concat([df, pd.get_dummies(df.STATUS, prefix='status')], axis=1).drop('STATUS', axis=1)
    buro_counts = df[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count()
    df['buro_count'] = df['SK_ID_BUREAU'].map(buro_counts['MONTHS_BALANCE'])
    avg_buro_bal = df.groupby('SK_ID_BUREAU').mean()
    avg_buro_bal.columns = ['avg_bb_' + f_ for f_ in avg_buro_bal.columns]
    return avg_buro_bal

bureau_balance_df = bureau_bal_transform(bureau_balance_df)
bureau_balance_df = to_float32(bureau_balance_df)
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
bureau_df = to_float32(bureau_df)
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
prev_app_df = pd.read_csv(os.path.join(input_dir, 'previous_application.csv'), nrows=sample_size*MULT)
##TODO the DAYS_ columns have too many 365243 values
def prevapp_transform(df):
    printlog('Process previous applications...')
    days_feat = ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"]
    for f_d in days_feat:
        v365243 = df[f_d] == 365243.000000
        df["is365243_" + f_d] = v365243.astype(np.float32)
    prev_cat_features = [f_ for f_ in df.columns if df[f_].dtype == 'object']
    prev_dum = pd.DataFrame()
    for f_ in prev_cat_features:
        dum = pd.get_dummies(df[f_], prefix=f_).astype(np.float32)
        prev_dum = pd.concat([prev_dum, dum], axis=1)
        del df[f_]
    df = pd.concat([df, prev_dum], axis=1)
    del prev_dum, v365243
    gc.collect()
    nb_prev_per_curr = df[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    df['SK_ID_PREV_cnt'] = df['SK_ID_CURR'].map(nb_prev_per_curr['SK_ID_PREV']).astype(np.float32)
    df = df.groupby('SK_ID_CURR').mean() #High Memory Usage
    return df
prev_app_df = to_float32(prev_app_df)
gc.collect()
prev_app_df = prevapp_transform(prev_app_df)
gc.collect()
printlog(prev_app_df.dtypes, "INFOBOX")
#%%
pos_cash_df = pd.read_csv(os.path.join(input_dir, 'POS_CASH_balance.csv'), nrows=sample_size*MULT)
def poscash_transform(df):
    printlog('Process POS_CASH...')
    df = pd.concat([df, pd.get_dummies(df['NAME_CONTRACT_STATUS'])], axis=1)
    nb_prevs = df[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    df['SK_ID_PREV_cnt'] = df['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV']).astype(np.float32)
    df = df.groupby('SK_ID_CURR').mean()
    return df

pos_cash_df = to_float32(pos_cash_df)
pos_cash_df = poscash_transform(pos_cash_df)
gc.collect()
printlog(pos_cash_df.head(), "INFOBOX")
#%%
credit_card_df = pd.read_csv(os.path.join(input_dir, 'credit_card_balance.csv'), nrows=sample_size*MULT)
def creditcard_transform(df):
    printlog('Process CREDIT CARD...')
    df = pd.concat([df, pd.get_dummies(df['NAME_CONTRACT_STATUS'], prefix='status_')], axis=1)
    nb_prevs = df[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    df['SK_ID_PREV_cnt'] = df['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV']).astype(np.float32)
    avg_cc_bal = df.groupby('SK_ID_CURR').mean()
    avg_cc_bal.columns = ['cc_' + f_ for f_ in avg_cc_bal.columns]
    return avg_cc_bal

credit_card_df = to_float32(credit_card_df)
credit_card_df = creditcard_transform(credit_card_df)
gc.collect()
printlog(credit_card_df.head(), "INFOBOX")

#%%
install_df = pd.read_csv(os.path.join(input_dir, 'installments_payments.csv'), nrows=sample_size*MULT)
def installm_transform(df):
    printlog('Process INSTALLMENT...')
    nb_prevs = df[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    df['SK_ID_PREV_cnt'] = df['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])
    avg_inst = df.groupby('SK_ID_CURR').mean()
    avg_inst.columns = ['inst_' + f_ for f_ in avg_inst.columns]    
    return avg_inst

install_df = to_float32(install_df)
install_df = installm_transform(install_df)
gc.collect()
printlog(install_df.head(), "INFOBOX")

#%%   
app_train_df = pd.read_csv(os.path.join(input_dir, 'application_train.csv'), nrows=sample_size)
app_test_df = pd.read_csv(os.path.join(input_dir, 'application_test.csv'), nrows=sample_size)

y = app_train_df['TARGET']
del app_train_df['TARGET']

categorical_feats = [f for f in app_train_df.columns if app_train_df[f].dtype == 'object']

for f_ in categorical_feats:
    app_train_df[f_], indexer = pd.factorize(app_train_df[f_])
    app_test_df[f_] = indexer.get_indexer(app_test_df[f_])

printlog('Merging all together...')
app_train_df = app_train_df.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
app_test_df = app_test_df.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
del avg_buro
gc.collect()

app_train_df = app_train_df.merge(right=prev_app_df.reset_index(), how='left', on='SK_ID_CURR')
app_test_df = app_test_df.merge(right=prev_app_df.reset_index(), how='left', on='SK_ID_CURR')
del prev_app_df
gc.collect()

app_train_df = app_train_df.merge(right=pos_cash_df.reset_index(), how='left', on='SK_ID_CURR')
app_test_df = app_test_df.merge(right=pos_cash_df.reset_index(), how='left', on='SK_ID_CURR')

app_train_df = app_train_df.merge(right=credit_card_df.reset_index(), how='left', on='SK_ID_CURR')
app_test_df = app_test_df.merge(right=credit_card_df.reset_index(), how='left', on='SK_ID_CURR')

app_train_df = app_train_df.merge(right=install_df.reset_index(), how='left', on='SK_ID_CURR')
app_test_df = app_test_df.merge(right=install_df.reset_index(), how='left', on='SK_ID_CURR')

del pos_cash_df, credit_card_df, install_df
gc.collect()

print('Shapes : ', app_train_df.shape, app_test_df.shape)

#%%