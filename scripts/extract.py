import pandas as pd
import json
import os
from tqdm import tqdm

COLUMN_NAMES = ['id', 'text', 'label']

def facebook_extractor():
    ROOT_URL = '../data/facebook'
    TRAIN_FILENAME = 'train.jsonl'
    TEST_FILENAME = 'test.jsonl'
    VAL_FILENAME = 'dev.jsonl'

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
            text = result.get('text')
            if index == 0:
                train_df.loc[len(train_df.index)] = [id, text, label]
            elif index == 1:
                test_df.loc[len(test_df.index)] = [id, text, label]
            else:
                val_df.loc[len(val_df.index)] = [id, text, label]

    train_df.to_csv(os.path.join(ROOT_URL, 'train.csv'))
    test_df.to_csv(os.path.join(ROOT_URL, 'test.csv'))
    val_df.to_csv(os.path.join(ROOT_URL, 'val.csv'))
    os.system("rm -rf " + os.path.join(ROOT_URL, TRAIN_FILENAME))
    os.system("rm -rf " + os.path.join(ROOT_URL, TEST_FILENAME))
    os.system("rm -rf " + os.path.join(ROOT_URL, VAL_FILENAME))

def mmhs150k_extractor():
    SPLIT_ROOT_URL = '../data/MMHS150K/splits' # Extract ids from text files
    TEXT_ROOT_URL = '../data/MMHS150K/img_txt' # Extract text using ids!
    ROOT_URL = '../data/MMHS150K'
    TRAIN_FILENAME = 'train_ids.txt'
    TEST_FILENAME = 'test_ids.txt'
    VAL_FILENAME = 'val_ids.txt'
    LABEL_FILE = '../data/MMHS150K/MMHS150K_GT.json' # Extract label using ids
    

    train_df = pd.DataFrame(columns=COLUMN_NAMES)
    test_df = pd.DataFrame(columns=COLUMN_NAMES)
    val_df = pd.DataFrame(columns=COLUMN_NAMES)

    split_train = open(os.path.join(SPLIT_ROOT_URL, TRAIN_FILENAME))
    split_test = open(os.path.join(SPLIT_ROOT_URL, TEST_FILENAME))
    split_val = open(os.path.join(SPLIT_ROOT_URL, VAL_FILENAME))

    

def main():
    # facebook_extractor()
    mmhs150k_extractor()

main()