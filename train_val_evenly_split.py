import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import glob

# 라벨이 저장된 텍스트 파일 경로

root_train_img_path = './dataset/neu-det/train/traindataset/images/'
root_train_label_path = './dataset/neu-det/train/traindataset/labels/'


train_img_path = './dataset/neu-det/train/traindataset/images/*.jpg'
train_label_path = './dataset/neu-det/train/traindataset/labels/*.txt'

# valid_img_path = './dataset/neu-det/val/images'
# valid_label_path = './dataset/neu-det/val/labels'


valid_img_path = './dataset/neu-det/trash'
valid_label_path = './dataset/neu-det/trash'

img_list = sorted(glob.glob(train_img_path))
label_list = sorted(glob.glob(train_label_path))
print(label_list)

class_count = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[]}
split_list = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[]}
count = 0

for i in range(len(label_list)):
    with open(label_list[i], 'r') as file:
        line = file.readline().strip()
        line = line.split(" ")[0]
        label_name = label_list[i].split('/')
        label_name = label_name[-1].split('.')[0]
        label_name = label_name.split('\\')[-1]        
        class_count[line].append(label_name)
        
for j in range(len(class_count)):
        # split_list[str(j)] = class_count[str(j)][int(len(class_count[str(j)])*0.6):int(len(class_count[str(j)])*0.8)]
        split_list[str(j)] = class_count[str(j)][int(len(class_count[str(j)])*0.8):]
        
for k in range(len(split_list)):
    count += len(split_list[str(k)])

print(split_list['1'])    


for i in range(len(split_list)):
    for k in range(len(split_list[str(i)])):
            img_path = os.path.join(root_train_img_path, split_list[str(i)][k])
            label_path = os.path.join(root_train_label_path, split_list[str(i)][k])
            print(img_path, label_path ,valid_img_path)
            shutil.move(img_path+'.jpg', valid_img_path)
            shutil.move(label_path+'.txt', valid_label_path)
    print(f'{i} / {len(split_list)}')
