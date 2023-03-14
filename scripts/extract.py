import pandas as pd
import json
import os
from tqdm import tqdm

ROOT_URL = '../data/facebook'
TRAIN_FILENAME = 'train.jsonl'
TEST_FILENAME = 'test.jsonl'
VAL_FILENAME = 'dev.jsonl'
COLUMN_NAMES = ['id', 'label']

train = open(os.path.join(ROOT_URL, TRAIN_FILENAME))
test = open(os.path.join(ROOT_URL, TEST_FILENAME))
val = open(os.path.join(ROOT_URL, VAL_FILENAME))

train_df = pd.DataFrame(columns=COLUMN_NAMES)
test_df = pd.DataFrame(columns=COLUMN_NAMES)
val_df = pd.DataFrame(columns=COLUMN_NAMES)

for index, image_set in enumerate([train, test, val]):
    for train_img in tqdm(image_set):
        result = json.loads(train_img)
        id = result.get('id')
        label = result.get('label')
        if index == 0:
            train_df.loc[len(train_df.index)] = [id, label]
        elif index == 1:
            test_df.loc[len(test_df.index)] = [id, label]
        else:
            val_df.loc[len(val_df.index)] = [id, label]

train_df.to_csv(os.path.join(ROOT_URL, 'train.csv'))
test_df.to_csv(os.path.join(ROOT_URL, 'test.csv'))
val_df.to_csv(os.path.join(ROOT_URL, 'val.csv'))
os.system("rm -rf " + os.path.join(ROOT_URL, TRAIN_FILENAME))
os.system("rm -rf " + os.path.join(ROOT_URL, TEST_FILENAME))
os.system("rm -rf " + os.path.join(ROOT_URL, VAL_FILENAME))