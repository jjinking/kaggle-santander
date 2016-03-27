'''
Description: Library of functions to be used in the Santander competition
Author: Joseph Kim
'''
import pandas as pd
import sys
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from os.path import expanduser
sys.path.insert(1, '{}/datsci'.format(expanduser('~')))

from datsci import eda, munge


FILE_TRAIN                                 = 'data/train.csv'
FILE_TRAIN_DEDUP                           = 'data/train.dedup.csv'
FILE_TRAIN_DEDUP_ONEHOT                    = 'data/train.dedup.onehot.csv'
FILE_TRAIN_DEDUP_ONEHOT_NA                 = 'data/train.dedup.onehot.na.csv'
FILE_TRAIN_DEDUP_ONEHOT_NA_IMPUTE_MEAN     = 'data/train.dedup.onehot.na.impute_mean.csv'
FILE_TRAIN_DEDUP_ONEHOT_NA_IMPUTE_MEDIAN   = 'data/train.dedup.onehot.na.impute_median.csv'
FILE_TRAIN_DEDUP_ONEHOT_NA_IMPUTE_FREQ     = 'data/train.dedup.onehot.na.impute_freq.csv'
FILE_TRAIN_DEDUP_ONEHOT_NA_ONEHOTINT       = 'data/train.dedup.onehot.na.onehotint.csv'

FILE_TEST                                  = 'data/test.csv'
FILE_TEST_DEDUP                            = 'data/test.dedup.csv'
FILE_TEST_DEDUP_ONEHOT                     = 'data/test.dedup.onehot.csv'
FILE_TEST_DEDUP_ONEHOT_NA                  = 'data/test.dedup.onehot.na.csv'
FILE_TEST_DEDUP_ONEHOT_NA_IMPUTE_MEAN      = 'data/test.dedup.onehot.na.impute_mean.csv'
FILE_TEST_DEDUP_ONEHOT_NA_IMPUTE_MEDIAN    = 'data/test.dedup.onehot.na.impute_median.csv'
FILE_TEST_DEDUP_ONEHOT_NA_IMPUTE_FREQ      = 'data/test.dedup.onehot.na.impute_freq.csv'
FILE_TEST_DEDUP_ONEHOT_NA_ONEHOTINT        = 'data/test.dedup.onehot.na.onehotint.csv'

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


def read_split(train_csv, test_csv):
    '''
    Read in csv files, and split the train data into train and test sets
    to be used during modeling.
    The test csv data is used for submissions
    '''
    df_train, df_test, feature_cols = read_data(train_csv, test_csv)

    # Split up the data
    X_all = df_train[feature_cols]  # feature values for all students
    y_all = df_train[TARGET_COL]

    test_size = 0.3 # 30 percent
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=0, stratify=y_all)

    return X_train, y_train, X_test, y_test, feature_cols, df_test


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


def one_hot_int(df_train, df_test, feature_cols):
    '''
    One-hot encode integer columns that only have a few unique values
    '''
    # Ignore already-one hot encoded columns
    int_cols = feature_cols[:]
    for c in feature_cols:
        if c[:6] == 'onehot':
            int_cols.remove(c)

    # Fine categorical columns
    categorical_cols = eda.find_categorical_columns(
        df_train[int_cols], df_test)

    # Convert non-null value containing columns to integers
    bad_cols = {'delta_imp_trasp_var17_in_1y3', 'delta_imp_trasp_var33_in_1y3'}
    for c, n in categorical_cols:
        # Dont turn null values to int
        if c not in bad_cols:
            df_train[c] = df_train[c].values.astype(int)

    # One-hot encode the categorical columns
    catcols = list(map(lambda t: t[0], categorical_cols))
    df_train = munge.one_hot_encode_features(df_train, columns=catcols)
    df_test = munge.one_hot_encode_features(df_test, columns=catcols)

    # Remove duplicate columns and rows
    df_train, df_test = remove_duplicates_const(df_train, df_test)

    return df_train, df_test