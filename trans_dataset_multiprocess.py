import glob
from PIL import Image,ImageDraw
import json
from threading import Thread
import time
from multiprocessing import Process,Lock
# ImageFile.LOAD_TRUNCATED_IMAGES = True
import xml.etree.ElementTree as ET
import cv2, os
import numpy as np
import torch,random


def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들 의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다 
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정
seed_everything(42)





def trans_dataset_json(img_path_list, label_path_list, irange ,resize=None):
    start_time = time.time()
    width,height = 0,0
    count = 0
    counta = 0
    if irange[1] == 0:  
        img_path_list = img_path_list[irange[0]:]
        label_path_list = label_path_list[irange[0]:]
    else:
        img_path_list = img_path_list[irange[0]:irange[1]]
        label_path_list = label_path_list[irange[0]:irange[1]]
    
    for q in range(len(img_path_list)):
    # for q in range(1):
        # q += 26900
        img_name = img_path_list[q].split('/')
        img_name = img_name[-1].split('.')[0]
        img_name = img_name.split('\\')[-1]
        # lock.acquire()
        try: 
            img = Image.open(img_path_list[q])
            width,height = img.size
            try:
                img = img.resize(resize)
                img.save(f'./dataset/fish/train/traindataset/images/{img_name}.jpg')
            except:
                with open('./dataset/log.txt', 'a', encoding='utf-8') as f:
                    f.write(f'{img_name}\n')
        finally:
            # lock.release()
            print(f"{q} / {len(img_path_list)}")    
    
    for i in range(len(label_path_list)):        
        # 라벨 이름 추출
        label_name = label_path_list[i].split('/')
        label_name = label_name[-1].split('.')[0]
        label_name = label_name.split('\\')[-1]
        with open('./dataset/log.txt','r',encoding='utf-8') as f:
            lines = [line.rstrip() for line in f]
        if label_name in lines:
            continue
        # json -> dictionary
        with open(label_path_list[i], 'r', encoding='utf-8') as f:
            label = json.load(f)
        width,height = label['images'][0]['width'],label['images'][0]['height']
        
        # label 저장 부분 x1 y1 x2 y2 
        if len(label['annotations']) == 0:
            with open(f'./dataset/fish/train/traindataset/labels/{label_name}.txt','w', encoding='utf-8') as file:
                file.write(" ") 
            with open('./dataset/log_none.txt', 'a', encoding='utf-8') as f:
                f.write(f'{label_name}\n')
        # lock.acquire()
    
        for j in range(len(label['annotations'])):
            if j == 0:
                counta += 1
            try:
                x = (label['annotations'][j]['bbox'][0] + label['annotations'][j]['bbox'][2]) /2.0 
                y = (label['annotations'][j]['bbox'][1] + label['annotations'][j]['bbox'][3]) /2.0
                w = label['annotations'][j]['bbox'][2] - label['annotations'][j]['bbox'][0]
                h = label['annotations'][j]['bbox'][3] - label['annotations'][j]['bbox'][1]
            # print(x,y,w,h)
            # print(width,height)
                x =  round((x / resize[0]) * (resize[0]/width),8)
                y =  round((y / resize[1])* (resize[1]/height),8)
                w =  round((w / resize[0]) * (resize[0]/width),8)
                h =  round((h  / resize[1])* (resize[1]/height),8)
            except:
                x,y,w,h = None,None,None,None
            if j == 0:
                with open(f'./dataset/fish/train/traindataset/labels/{label_name}.txt','w', encoding='utf-8') as file:
                    count += 1
                    try: 
                        file.write(f"{label['annotations'][j]['symptom']-1} {x} {y} {w} {h}\n")
                        print(f'{i} / {len(label_path_list)}')
                    except:
                        file.write(" ")
                        
            else:
                with open(f'./dataset/fish/train/traindataset/labels/{label_name}.txt','a', encoding='utf-8') as file:
                    file.write(f"{label['annotations'][j]['symptom']-1} {x} {y} {w} {h}\n")
    
            # lock.release()
    print("총 라벨 수:", count)
    end_time = time.time()
    print(end_time - start_time)  
    
def trans_dataset_xml(img_path_list, label_path_list, irange ,resize=None):
    with open('./dataset/class_list.json','r',encoding='utf-8') as f:
        class_list = json.load(f)
    start_time = time.time()
    width,height = 0,0
    count = 0
    img_name_list = []
    
    
    if irange[1] == 0:  
        img_path_list = img_path_list[irange[0]:]
        label_path_list = label_path_list[irange[0]:]
    else:
        img_path_list = img_path_list[irange[0]:irange[1]]
        label_path_list = label_path_list[irange[0]:irange[1]]
    for q in range(len(img_path_list)):
    # for q in range(1):
        # q += 26900
        img_name = img_path_list[q].split('/')
        
        img_name = img_name[-1].split('.')[0]
        img_name = img_name.split('\\')[-1]

    # lock.acquire()
    # try: 
        img = Image.open(img_path_list[q])
        width,height = img.size
    # try:
        img = img.resize(resize)
        # 이미지 sharpness 추가
        img = sharpness_algo(img)
        img.save(f'./dataset/neu-det/train/traindataset/images/{img_name}.jpg')
        img_name_list.append(img_name)
        # except:
    #         with open('./dataset/log.txt', 'a', encoding='utf-8') as f:
    #             f.write(f'{img_name}\n')
    # finally:
    #     # lock.release()
        # print(f"{q} / {len(img_path_list)}")    

    for i in range(len(label_path_list)):
        # 라벨 이름 추출
        label_name = label_path_list[i].split('/')
        label_name = label_name[-1].split('.')[0]
        label_name = label_name.split('\\')[-1]
        with open('./dataset/log.txt','r',encoding='utf-8') as f:
            lines = [line.rstrip() for line in f]
        if label_name in lines:
            continue
        # json -> dictionary
        tree = ET.parse(label_path_list[i])
        root = tree.getroot()
        width,height = int(root.find('size').find('width').text),int(root.find('size').find('height').text)
        # print(width,height)
        
        # label 저장 부분 x1 y1 x2 y2 
        ocount = 0
        for i in root.iter('object'): ocount += 1
        if ocount == 0:
            with open(f'./dataset/neu-det/train/traindataset/labels/{label_name}.txt','w', encoding='utf-8') as file:
                file.write(" ") 
            with open('./dataset/log_none.txt', 'a', encoding='utf-8') as f:
                f.write(f'{label_name}\n')
        # lock.acquire()
        first = 0
        for obj in root.iter('object'):
            first += 1
            try:
                x = (float(obj.find('bndbox').find('xmin').text) + float(obj.find('bndbox').find('xmax').text)) /2.0 
                y = (float(obj.find('bndbox').find('ymin').text) + float(obj.find('bndbox').find('ymax').text)) /2.0
                w = float(obj.find('bndbox').find('xmax').text) - float(obj.find('bndbox').find('xmin').text)
                h = float(obj.find('bndbox').find('ymax').text) - float(obj.find('bndbox').find('ymin').text)
                # print(x,y,w,h)
                # print(width,height)
                x =  round((x / resize[0]) * (resize[0]/width),8)
                y =  round((y / resize[1])* (resize[1]/height),8)
                w =  round((w / resize[0]) * (resize[0]/width),8)
                h =  round((h  / resize[1])* (resize[1]/height),8)
            except:
                x,y,w,h = None,None,None,None
            if first == 1:
                with open(f'./dataset/neu-det/train/traindataset/labels/{label_name}.txt','w', encoding='utf-8') as file:
                    count += 1
                    try: 
                        file.write(f"{class_list[obj.find('name').text]} {x} {y} {w} {h}\n")
                        # print(f'{i} / {len(label_path_list)}')
                    except:
                        file.write(" ")
                        
            else:
                with open(f'./dataset/neu-det/train/traindataset/labels/{label_name}.txt','a', encoding='utf-8') as file:
                        file.write(f"{class_list[obj.find('name').text]} {x} {y} {w} {h}\n")
    
            # lock.release()
    #augmentation 수행
    augmentation(img_path_list, label_path_list,img_name_list, class_list,resize)
    print("총 라벨 수:", count)
    end_time = time.time()
    print(end_time - start_time)

def sharpness_algo(img):
    img = np.array(img)
      
    src_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    src_f = src_ycrcb[:, :, 0].astype(np.float32)
    blr = cv2.GaussianBlur(src_f, (0, 0), 2.0)
    src_ycrcb[:, :, 0] = np.clip(2. * src_f - blr, 0, 255).astype(np.uint8)
    dst = cv2.cvtColor(src_ycrcb, cv2.COLOR_YCrCb2RGB)
    
    sharp_img = Image.fromarray(dst)
    return sharp_img

def flip(img, bbox, img_name, class_list,resize):
    width, height = img.size  
    img = img.resize(resize)
    # img = sharpness_algo(img)  
    flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    flip_img.save(f'./dataset/neu-det/train/traindataset/images/{img_name}_1.jpg')
    
    root = bbox.getroot()
    first = 0
    for obj in root.iter('object'):
        first += 1
        try:
            x_min = width - float(obj.find('bndbox').find('xmax').text)
            y_min = float(obj.find('bndbox').find('ymin').text)
            x_max = width - float(obj.find('bndbox').find('xmin').text)
            y_max = float(obj.find('bndbox').find('ymax').text)
            x = (x_min + x_max) /2.0 
            y = (y_min + y_max) /2.0
            w = x_max - x_min
            h = y_max - y_min
            x =  round((x/resize[0]) * (resize[0]/width),8)
            y =  round((y/resize[1])* (resize[1]/height),8)
            w =  round((w/resize[0]) * (resize[0]/width),8)
            h =  round((h/resize[1])* (resize[1]/height),8)
        except:
            x,y,w,h = None,None,None,None
        if first == 1:
            with open(f'./dataset/neu-det/train/traindataset/labels/{img_name}_1.txt','w', encoding='utf-8') as file:
                try: 
                    file.write(f"{class_list[obj.find('name').text]} {x} {y} {w} {h}\n")
                    # print(f'{i} / {len(label_path_list)}')
                except:
                    file.write(" ")        
        else:
            with open(f'./dataset/neu-det/train/traindataset/labels/{img_name}_1.txt','a', encoding='utf-8') as file:
                    file.write(f"{class_list[obj.find('name').text]} {x} {y} {w} {h}\n")



def cutout(img, bbox, img_name, class_list,resize):
        
    width, height = img.size  
    img = img.resize(resize)
    
    root = bbox.getroot()
    first = 0
    for obj in root.iter('object'):
        first += 1
        try:
            x_min = float(obj.find('bndbox').find('xmin').text)
            y_min = float(obj.find('bndbox').find('ymin').text)
            x_max = float(obj.find('bndbox').find('xmax').text)
            y_max = float(obj.find('bndbox').find('ymax').text)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            x = (float(obj.find('bndbox').find('xmin').text) + float(obj.find('bndbox').find('xmax').text)) /2.0 
            y = (float(obj.find('bndbox').find('ymin').text) + float(obj.find('bndbox').find('ymax').text)) /2.0
            w = float(obj.find('bndbox').find('xmax').text) - float(obj.find('bndbox').find('xmin').text)
            h = float(obj.find('bndbox').find('ymax').text) - float(obj.find('bndbox').find('ymin').text)
            # print(x,y,w,h)
            # print(width,height)
            x =  round((x / resize[0]) * (resize[0]/width),8)
            y =  round((y / resize[1])* (resize[1]/height),8)
            w =  round((w / resize[0]) * (resize[0]/width),8)
            h =  round((h  / resize[1])* (resize[1]/height),8)
            
        except:
            x,y,w,h = None,None,None,None
        draw = ImageDraw.Draw(img)
        square_width = bbox_width * 0.3
        square_height= bbox_height * 0.3
        
        start_x = np.random.randint(x_min, x_max)
        start_y = np.random.randint(y_min, y_max)
        print(start_x, start_y, start_x + square_width, start_y + square_height, "fill 박스")
        print(x_min,y_min,x_max,y_max, "원래 박스")
        print(square_width, square_height, "줄어든 크기")
        draw.rectangle((start_x*3.2, start_y*3.2, (start_x + square_width)*3.2, (start_y + square_height)*3.2), fill=(0,0,0))
        
        if first == 1:
            with open(f'./dataset/neu-det/train/traindataset/labels/{img_name}_2.txt','w', encoding='utf-8') as file:
                try: 
                    file.write(f"{class_list[obj.find('name').text]} {x} {y} {w} {h}\n")
                    # print(f'{i} / {len(label_path_list)}')
                except:
                    file.write(" ")        
        else:
            with open(f'./dataset/neu-det/train/traindataset/labels/{img_name}_2.txt','a', encoding='utf-8') as file:
                    file.write(f"{class_list[obj.find('name').text]} {x} {y} {w} {h}\n")
    img.save(f'./dataset/neu-det/train/traindataset/images/{img_name}_2.jpg')          
    
    

def augmentation(img_path_list, label_path_list, img_name_list, class_list,resize=None):
    for i in range(len(img_path_list)):
        img = Image.open(img_path_list[i])        
        tree = ET.parse(label_path_list[i])
        
        flip(img, tree, img_name_list[i],class_list,resize)
        cutout(img, tree, img_name_list[i],class_list,resize)
        
        
                                           
                    
if __name__ == '__main__':
        
    label_path = './dataset/neu-det/train/origin_labels/*.xml'
    label_path_list = sorted(glob.glob(label_path))
    
    
    img_path = './dataset/neu-det/train/origin_images/*.jpg'
    img_path_list = sorted(glob.glob(img_path))
    resize = (640,640)
    # resize= None
    
    multi = 23
    
    z = int(len(img_path_list) / multi)
    processes = []
    
    
    for i in range(multi):
        irange = (z*i, z*(i+1))
        if i == multi-1:
            irange = (z*i,0)
        process = Process(target=trans_dataset_xml, args=(img_path_list, label_path_list, irange,resize), name=f'Porcess-{i+1}')
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
        
    print("모든 프로세스가 종료됨")
    
    
    
    
    # trans_dataset(img_path_list,label_path_list,resize)