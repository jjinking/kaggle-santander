'''
Description: Library of functions to be used in the Santander competition
Author: Joseph Kim
'''
import pandas as pd
import sys
from sklearn.cross_validation import train_test_split
from os.path import expanduser
sys.path.insert(1, '{}/datsci'.format(expanduser('~')))

from datsci import eda, munge


TARGET_COL = 'TARGET'


def read_data(train_csv, test_csv):
    '''
    Read in csv files
    '''
    # Read in data
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


def csv_remove_duplicates_const(train_in_csv, test_in_csv,
                                train_out_csv, test_out_csv):
    '''
    Read in csv files and remove duplicate and const cols
    Also remove duplicate rows in train data
    Save results to file
    '''
    df_train, df_test, feature_cols = read_data(train_in_csv, test_in_csv)
    df_train, df_test = remove_duplicates_const(df_train, df_test)
    write_data(df_train, df_test, train_out_csv, test_out_csv)


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


def csv_one_hot_encode_binary_features(train_in_csv, test_in_csv,
                                       train_out_csv, test_out_csv):
    '''
    Read in csv files and one-hot encode the binary features,
    which start with "ind_"
    and save results to file
    '''
    df_train, df_test, feature_cols = read_data(train_in_csv, test_in_csv)
    df_train, df_test = one_hot_encode_binary_features(df_train, df_test)
    write_data(df_train, df_test, train_out_csv, test_out_csv)
