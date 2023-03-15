import PIL
import os
import pandas as pd
from tqdm import tqdm

ROOT_URL = '../data/facebook/img'

train_df = pd.read_csv('../data/facebook/train.csv')
test_df = pd.read_csv('../data/facebook/test.csv')
val_df = pd.read_csv('../data/facebook/val.csv')

def strip_names(path):
    for fileName in os.listdir(path):
        modified_fileName = ""
        for index, char in enumerate(fileName):
            if char != '0':
                break
            else:
                modified_fileName = fileName[index + 1:]
        os.system('mv ' + path + '/' + fileName + ' ' + path + '/' + modified_fileName)

def segregate_images():
    os.system('mkdir ' + os.path.join(ROOT_URL) + '/train/')
    os.system('mkdir ' + os.path.join(ROOT_URL) + '/test/')
    os.system('mkdir ' + os.path.join(ROOT_URL) + '/val/')
    for index, df in enumerate([train_df, test_df, val_df]):
        for id in tqdm(df['id']):
            if index == 0:
                os.system('mv ' + ROOT_URL + '/' + str(id) + '.png' + ' ' + ROOT_URL + '/train')
            elif index == 1:
                os.system('mv ' + ROOT_URL + '/' + str(id) + '.png' + ' ' + ROOT_URL + '/test')
            else:
                os.system('mv ' + ROOT_URL + '/' + str(id) + '.png' + ' ' + ROOT_URL + '/val')

def main():
    strip_names(ROOT_URL)
    segregate_images()

main()