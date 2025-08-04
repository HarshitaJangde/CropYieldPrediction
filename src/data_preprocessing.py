import pandas as pd

def preprocess_data(df):
    feature_cols = ['Fertilizer', 'temp', 'N', 'P', 'K']
    X = df[feature_cols]
    y = df['yeild']
    return X, y
