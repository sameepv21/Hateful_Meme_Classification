import PIL
import os
import pandas as pd
from tqdm import tqdm

ROOT_URL = '../data/facebook'

train_df = pd.read_json('../data/facebook/train.jsonl', lines = True)
test_df = pd.read_json('../data/facebook/test.jsonl', lines = True)
dev_df = pd.read_json('../data/facebook/dev.jsonl', lines = True)

def segregate_images():
    os.system('mkdir ' + os.path.join(ROOT_URL) + '/train/')
    os.system('mkdir ' + os.path.join(ROOT_URL) + '/test/')
    os.system('mkdir ' + os.path.join(ROOT_URL) + '/dev/')
    for index, df in enumerate([train_df, test_df, dev_df]):
        for img in tqdm(df['img']):
            if index == 0:
                os.system('mv ' + ROOT_URL + '/' + img + ' ' + ROOT_URL + '/train')
                new_path = "train/" + img.strip('img/') + 'g'
            elif index == 1:
                os.system('mv ' + ROOT_URL + '/' + img + ' ' + ROOT_URL + '/test')
                new_path = "test/" + img.strip('img/') + 'g'
            else:
                os.system('mv ' + ROOT_URL + '/' + img + ' ' + ROOT_URL + '/dev')
                new_path = "dev/" + img.strip('img/') + 'g'
            df['img'] = df['img'].replace(img, new_path)
        out = df.to_json(orient='records')[1:-1].replace('},{', '} {')
        print(out)
        if(index == 0):
            with open('../data/facebook/train.jsonl') as f:
                f.write(out)
        elif(index == 1):
            with open('../data/facebook/test.jsonl') as f:
                f.write(out)
        else:
            with open('../data/facebook/dev.jsonl') as f:
                f.write(out)

def separate_classes():
    os.system("mkdir " + os.path.join(ROOT_URL) + '/train/hateful')
    os.system("mkdir " + os.path.join(ROOT_URL) + '/train/non_hateful')
    os.system("mkdir " + os.path.join(ROOT_URL) + '/dev/hateful')
    os.system("mkdir " + os.path.join(ROOT_URL) + '/dev/non_hateful')

    for index, df in enumerate([train_df, dev_df]):
        for img in tqdm(df['img']):
            label = df[df['id'] == id].iloc[0]['label']
            if index == 0:
                if label == 0:
                    os.system('mv ' + ROOT_URL + '/' + img + ' ' + ROOT_URL + '/train/non_hateful')
                    new_path = "train/non_hateful/" + img.strip('train/')
                else:
                    os.system('mv ' + ROOT_URL + '/train/' + img + ' ' + ROOT_URL + '/train/hateful')
                    new_path = "train/hateful/" + img.strip('train/')
            else:
                if label == 0:
                    os.system('mv ' + ROOT_URL + '/dev/' + img + ' ' + ROOT_URL + '/dev/non_hateful')
                    new_path = "dev/non_hateful/" + img.strip('dev/')
                else:
                    os.system('mv ' + ROOT_URL + '/dev/' + img + ' ' + ROOT_URL + '/dev/hateful')
                    new_path = "dev/hateful/" + img.strip('dev/')
            df['img'] = df['img'].replace(img, new_path)
        out = df.to_json(orient='records')[1:-1].replace('},{', '} {')
        if(index == 0):
            with open('../data/facebook/train.jsonl') as f:
                f.write(out)
        elif(index == 1):
            with open('../data/facebook/test.jsonl') as f:
                f.write(out)
        else:
            with open('../data/facebook/dev.jsonl') as f:
                f.write(out)

def main():
    segregate_images()
    separate_classes()
main()