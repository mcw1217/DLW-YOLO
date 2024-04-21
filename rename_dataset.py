import glob
from PIL import Image,ImageFile
import json
from threading import Thread
import time
from multiprocessing import Process,Lock
import xml.etree.ElementTree as ET
import cv2, os
import numpy as np


if __name__ == '__main__':
        
    label_path = './dataset/neu-det/train/origin_labels/*.xml'
    label_path_list = sorted(glob.glob(label_path))
    
    
    img_path = './dataset/neu-det/train/origin_images/*.jpg'
    img_path_list = sorted(glob.glob(img_path))
    
    for i in range(len(img_path_list)):
        img_name = img_path_list[i].split('/')
        
        img_name = img_name[-1].split('.')[0]
        img_name = img_name.split('\\')[-1]
        
        img_rename = img_name.split('_')
        
        if len(img_rename) == 3:
            img_rename = [img_rename[0]+"_"+img_rename[1], img_rename[2]]
        
        if len(img_rename[1]) == 1:
            img_rename = img_rename[0] + '_00' + img_rename[1]
        elif len(img_rename[1]) == 2:
            img_rename = img_rename[0] + '_0' + img_rename[1]
        elif len(img_rename[1]) == 3:
            img_rename = img_rename[0] + '_' + img_rename[1]
        
        print(img_rename)
        os.rename(img_path_list[i],f'./dataset/neu-det/train/origin_images/{img_rename}.jpg')
        os.rename(label_path_list[i],f'./dataset/neu-det/train/origin_labels/{img_rename}.xml')
            
    
