'''
Description: Library of functions to be used in the Santander competition
Author: Joseph Kim
'''

import numpy as np
import pandas as pd
import sys
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer

from os.path import expanduser
sys.path.insert(1, '{}/datsci'.format(expanduser('~')))
from datsci import eda, munge


FILE_TRAIN                                         = 'data/train.csv'
FILE_TRAIN_DEDUP                                   = 'data/train.dedup.csv'
# FILE_TRAIN_DEDUP_ONEHOT                            = 'data/train.dedup.onehot.csv'
# FILE_TRAIN_DEDUP_ONEHOT_NA                         = 'data/train.dedup.onehot.na.csv'
# FILE_TRAIN_DEDUP_ONEHOT_NA_IMPUTE_MEAN             = 'data/train.dedup.onehot.na.impute_mean.csv'
# FILE_TRAIN_DEDUP_ONEHOT_NA_IMPUTE_MEDIAN           = 'data/train.dedup.onehot.na.impute_median.csv'
# FILE_TRAIN_DEDUP_ONEHOT_NA_IMPUTE_FREQ             = 'data/train.dedup.onehot.na.impute_freq.csv'
# FILE_TRAIN_DEDUP_ONEHOT_NA_ONEHOTINT               = 'data/train.dedup.onehot.na.onehotint.csv'
FILE_TRAIN_DEDUP_VAR3_DELTA1_1HOT                  = 'data/train.dedup.var3.delta1.1hot.csv'
FILE_TRAIN_DEDUP_VAR3_DELTA1_1HOT_1HOTINT          = 'data/train.dedup.var3.delta1.1hot.1hotint.csv'
FILE_TRAIN_DEDUP_VAR3_DELTANAN_1HOT                = 'data/train.dedup.var3.deltanan.1hot.csv'
FILE_TRAIN_DEDUP_VAR3_DELTANAN_1HOT_1HOTINT        = 'data/train.dedup.var3.deltanan.1hot.1hotint.csv'

FILE_TEST                                          = 'data/test.csv'
FILE_TEST_DEDUP                                    = 'data/test.dedup.csv'
# FILE_TEST_DEDUP_ONEHOT                             = 'data/test.dedup.onehot.csv'
# FILE_TEST_DEDUP_ONEHOT_NA                          = 'data/test.dedup.onehot.na.csv'
# FILE_TEST_DEDUP_ONEHOT_NA_IMPUTE_MEAN              = 'data/test.dedup.onehot.na.impute_mean.csv'
# FILE_TEST_DEDUP_ONEHOT_NA_IMPUTE_MEDIAN            = 'data/test.dedup.onehot.na.impute_median.csv'
# FILE_TEST_DEDUP_ONEHOT_NA_IMPUTE_FREQ              = 'data/test.dedup.onehot.na.impute_freq.csv'
# FILE_TEST_DEDUP_ONEHOT_NA_ONEHOTINT                = 'data/test.dedup.onehot.na.onehotint.csv'
FILE_TEST_DEDUP_VAR3_DELTA1_1HOT                   = 'data/test.dedup.var3.delta1.1hot.csv'
FILE_TEST_DEDUP_VAR3_DELTA1_1HOT_1HOTINT           = 'data/test.dedup.var3.delta1.1hot.1hotint.csv'
FILE_TEST_DEDUP_VAR3_DELTANAN_1HOT                 = 'data/test.dedup.var3.deltanan.1hot.csv'
FILE_TEST_DEDUP_VAR3_DELTANAN_1HOT_1HOTINT         = 'data/test.dedup.var3.deltanan.1hot.1hotint.csv'


FILE_SAMPLE_SUBMIT                         = 'data/sample_submission.csv'

TARGET_COL                                 = 'TARGET'


def read_data(train_csv, test_csv):
    '''
    Read in csv files
    '''
    df_train = pd.read_csv(train_csv, index_col='ID')
    feature_cols = list(df_train.columns)
    feature_cols.remove(TARGET_COL)
    df_test = pd.read_csv(test_csv, index_col='ID')
    return df_train, df_test, feature_cols


def write_data(df_train, df_test, train_csv, test_csv):
    '''
    Write DataFrame data to csv file
    '''
    df_train.to_csv(train_csv)
    df_test.to_csv(test_csv)


def read_process_write(train_in_csv, test_in_csv,
                       train_out_csv, test_out_csv,
                       process_func,
                       pass_features=False,
                       process_kwargs={}):
    '''
    Read in data, process them, and save results to file
    '''
    df_train, df_test, feature_cols = read_data(train_in_csv, test_in_csv)
    if pass_features:
        df_train, df_test = process_func(
            df_train, df_test, feature_cols, **process_kwargs)
    else:
        df_train, df_test = process_func(df_train, df_test, **process_kwargs)
    write_data(df_train, df_test, train_out_csv, test_out_csv)


def summarize_files():
    '''
    Summarize number of rows and columns in data files
    '''
    def get_sizes(train_csv, test_csv):
        df_train, df_test, feature_cols = read_data(train_csv, test_csv)
        train_rows, train_cols = df_train.shape
        test_rows, test_cols = df_test.shape
        return train_rows, train_cols, test_rows, test_cols

    data_shapes = []
    for s, train_csv, test_csv in [
            ('raw',           FILE_TRAIN,                                 FILE_TEST),
            ('dedup',         FILE_TRAIN_DEDUP,                           FILE_TEST_DEDUP),
            ('bin onehot',    FILE_TRAIN_DEDUP_ONEHOT,                    FILE_TEST_DEDUP_ONEHOT),
            ('NaN',           FILE_TRAIN_DEDUP_ONEHOT_NA,                 FILE_TEST_DEDUP_ONEHOT_NA),
            ('impute mean',   FILE_TRAIN_DEDUP_ONEHOT_NA_IMPUTE_MEAN,     FILE_TEST_DEDUP_ONEHOT_NA_IMPUTE_MEAN),
            ('impute median', FILE_TRAIN_DEDUP_ONEHOT_NA_IMPUTE_MEDIAN,   FILE_TEST_DEDUP_ONEHOT_NA_IMPUTE_MEDIAN),
            ('impute freq',   FILE_TRAIN_DEDUP_ONEHOT_NA_IMPUTE_FREQ,     FILE_TEST_DEDUP_ONEHOT_NA_IMPUTE_FREQ),
            ('onehot int',    FILE_TRAIN_DEDUP_ONEHOT_NA_ONEHOTINT,       FILE_TEST_DEDUP_ONEHOT_NA_ONEHOTINT),
    ]:
        data_shapes.append((s,) + get_sizes(train_csv, test_csv))
    return pd.DataFrame(data_shapes, columns=['stage', 'train rows', 'train cols', 'test rows', 'test cols'])


def read_split(train_csv, test_csv, test_size=0.3, random_state=0):
    '''
    Read in csv files, and split the train data into train and test sets
    to be used during modeling.
    The test csv data is used for submissions
    '''
    df_train, df_test, feature_cols = read_data(train_csv, test_csv)

    # Split up the data
    X_all = df_train[feature_cols]
    y_all = df_train[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all)

    return X_train, y_train, X_test, y_test, feature_cols, df_train, df_test


def dedup(df):
    '''
    Remove duplicate rows and columns
    '''
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Remove duplicate columns
    return munge.remove_duplicates(df.T).T


def remove_duplicates_const(df_train, df_test):
    '''
    Remove duplicate rows and columns and constant columns
    '''
    df_train = dedup(df_train)

    # Remove constant columns
    df_train.drop(eda.find_const_cols(df_train), axis=1, inplace=True)

    # Remove duplicate rows
    df_train.drop_duplicates(inplace=True)

    feature_cols = list(df_train.columns)
    feature_cols.remove(TARGET_COL)
    df_test = df_test[feature_cols]

    return df_train, df_test


def one_hot_encode_binary_features(df_train, df_test):
    '''
    One-hot encode binary features, which start with "ind_"
    '''
    binary_cols = [c for c in df_train.columns if c[:4] == 'ind_']

    # Convert to int
    for c in binary_cols:
        df_train[c] = df_train[c].values.astype(int)

    df_train = munge.one_hot_encode_features(df_train, columns=binary_cols)
    df_test = munge.one_hot_encode_features(df_test, columns=binary_cols)

    # Remove duplicates, just in case
    df_train, df_test = remove_duplicates_const(df_train, df_test)

    return df_train, df_test


def impute_null_vals(df_train, df_test, feature_cols, strategy='mean'):
    '''
    Impute null values using strategy
    '''
    # Impute using combined (train + test) datasets
    df_combined = df_train[feature_cols].append(df_test[feature_cols])
    imputer = Imputer(
        missing_values='NaN', strategy=strategy, axis=0, verbose=0, copy=False
    ).fit(df_combined)
    df_train[feature_cols] = imputer.transform(df_train[feature_cols])
    df_test[feature_cols] = imputer.transform(df_test[feature_cols])

    # Remove duplicate columns and rows
    df_train, df_test = remove_duplicates_const(df_train, df_test)

    return df_train, df_test


def one_hot_int(df_train, df_test, feature_cols, delta_nulltype=int):
    '''
    One-hot encode integer columns that only have a few unique values

    delta_nulltype can be either int or np.nan, which determines what the
    anomaly values in delta_ columns have been replaced with.
    '''
    # Ignore already-one hot encoded columns
    int_cols = feature_cols[:]
    for c in feature_cols:
        if c[:6] == 'onehot':
            int_cols.remove(c)

    # Fine categorical columns
    categorical_cols = eda.find_categorical_columns(
        df_train[int_cols], df_test)

    # Convert df_train categorical columns to integers
    # Don't convert column containing float values to int
    bad_cols = {"delta_num_aport_var33_1y3"}  # delta_nulltype is int
    if delta_nulltype is np.nan:
        # Convert only non-null value containing columns to integers
        bad_cols = {'delta_imp_trasp_var17_in_1y3',
                    'delta_imp_trasp_var33_in_1y3'}
    for c, n in categorical_cols:
        if c not in bad_cols:
            df_train[c] = df_train[c].values.astype(int)

    # One-hot encode the categorical columns
    catcols = list(map(lambda t: t[0], categorical_cols))
    df_train = munge.one_hot_encode_features(df_train, columns=catcols)
    df_test = munge.one_hot_encode_features(df_test, columns=catcols)

    # Remove duplicate columns and rows
    df_train, df_test = remove_duplicates_const(df_train, df_test)

    return df_train, df_test


def set_var3_null(df_train, df_test):
    '''
    Replace the null values in var3 column
    '''
    df_train['var3'] = df_train.var3.replace(-999999, np.nan)
    df_test['var3'] = df_test.var3.replace(-999999, np.nan)
    return df_train, df_test


def fix_delta_cols(df_train, df_test, replace_with=1):
    '''
    Fix the columns starting with "delta", which contain
    9999999999 values
    '''
    ANOMALY = 9999999999
    # Find delta cols
    for c in df_train:
        if c.find('delta') == 0:
            df_train[c] = df_train[c].replace(ANOMALY, replace_with)
            df_test[c] = df_test[c].replace(ANOMALY, replace_with)
    return df_train, df_test


def cv_fit_xgb_model(model,
                     X_train, y_train,
                     X_test, y_test,
                     cv_nfold=5,
                     early_stopping_rounds=50,
                     missing=np.nan):
    '''
    Fit xgb model with best n_estimators using xgb builtin cv
    '''

    # Train cv
    xgb_param = model.get_xgb_params()
    dtrain = xgb.DMatrix(X_train.values, label=y_train.values, missing=missing)
    cv_result = xgb.cv(
        xgb_param,
        dtrain,
        num_boost_round=model.get_params()['n_estimators'],
        nfold=cv_nfold,
        metrics=['auc'],
        early_stopping_rounds=early_stopping_rounds,
        show_progress=False)
    best_n_estimators = cv_result.shape[0]
    model.set_params(n_estimators=best_n_estimators)

    # Train model
    model.fit(X_train, y_train, eval_metric='auc')

    # Predict training data
    y_hat_train = model.predict(X_train)

    # Predict test data
    y_hat_test = model.predict(X_test)

    # Print model report:
    print("\nModel Report")
    print("best n_estimators: {}".format(best_n_estimators))
    print("AUC Score (Train): %f" % roc_auc_score(y_train, y_hat_train))
    print("AUC Score (Test) : %f" % roc_auc_score(y_test,  y_hat_test))

    # feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importance Score')
