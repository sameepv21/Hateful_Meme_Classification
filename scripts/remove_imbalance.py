import pandas as pd
from imblearn.over_sampling import SMOTENC

def balance(path):
    df = pd.read_json(path, lines = True)

    y = df['label']
    x = df.drop('label', axis = 1)

    nm = SMOTENC(random_state = 42, categorical_features = [1,2])
    X_new, y_new = nm.fit_resample(x, y)
    df_new = pd.concat([X_new, y_new], axis = 1)
    df_new.to_json(path, orient = 'records')

balance('../data/facebook/train.jsonl')
balance('../data/facebook/dev.jsonl')