import pandas as pd
import logging
import csv
import numpy as np
import os

_INPUT_FILE = 'harddrive-smart-data-pp-to-train.csv'
_INPUT_DATA_FILE = 'harddrive-smart-data.csv'

_OUTPUT_PATH = "anomaly_scores.csv"

def convertorToFloat(val):
    if val == 'True':
        val = 1
    elif val == 'False':
        val = 0
    elif val is False:
        val = 0
    return val


# select feature set
def dataCleanser(inputFile):
    # Won't be converting final data frame to float here. Will be doing on the fly while training
    df = pd.read_csv(inputFile)

    print(df.shape)

    colsToDrop = ['GList1', 'PList', 'Servo1', 'Servo2', 'Servo3', 'Servo5',
                  'ReadError1', 'ReadError2', 'ReadError3', 'FlyHeight5',
                  'ReadError18', 'ReadError19', 'Servo7', 'Servo8', 'ReadError20', 'GList2',
                  'GList3', 'Servo10']
    df = df.drop(colsToDrop, axis=1)

    df['class'] = df['class'].apply(convertorToFloat)
    print("DF CLASS")
    print(df['class'])

    df.to_csv('harddrive-smart-data-temp.csv', sep=',', index=False)

    df = pd.read_csv('harddrive-smart-data-temp.csv')
    df.to_csv('harddrive-smart-data.csv', sep=',', index=False)



def get_train_test_inds(y,train_proportion=0.7):
    '''Generates indices, making random stratified split into training set and testing sets
    with proportions train_proportion and (1-train_proportion) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and
    testing sets are preserved (stratified sampling).
    '''

    y=np.array(y)
    y.setflags(write=True)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        value_inds.setflags(write=True)
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))

        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True

    return train_inds,test_inds


# split data into good and bad drives, also training and test data
def dataSplit():
    df = pd.read_csv('harddrive-smart-data.csv')
    dfBad = df.loc[df['class'] == 1.0]
    dfGood = df.loc[df['class'] == 0.0]

    dfBad.to_csv('harddrive-smart-data-bad.csv', sep=',', index=False)
    dfGood.to_csv('harddrive-smart-data-good.csv', sep=',', index=False)

    print(dfBad.shape)
    print(dfGood.shape)

if __name__ == '__main__':
    dataCleanser(_INPUT_FILE)

