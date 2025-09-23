import pandas as pd

def mirar_NA(df):
    h = df.isna().sum(axis = 0)
    return h[h>0]

def save_clean_data(df, file):
    df.to_csv(file, index=False, encoding='UTF-8')
    return None