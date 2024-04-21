from ultralytics import YOLO
import torch
import os
import numpy as np
import random, requests
import time
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"


api_url = "https://notify-api.line.me/api/notify"
token = "LYy0yPmrqjMc3rmvdQR2WcbCCVZkmFlf6FZBZGEkpYQ"

headers = {'Authorization':'Bearer '+token}


def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들 의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다 
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정
seed_everything(42)


if __name__ == '__main__':
        # model = YOLO("ultralytics/cfg/models/v8/yolov8n-ca.yaml")
        model = YOLO("yolov8n.yaml")
        # model = YOLO("yolov8n.pt")
        # model = YOLO("./runs/detect/applied_sharp/weights/best.pt")
        s = time.time()
        message = {
                    f"message" : "[집 컴퓨터] : 모델 학습이 시작되었습니다!"
                  }
        requests.post(api_url, headers= headers , data = message)
        # with torch.autocast("cuda"):
        model.train(data="data.yaml", epochs=10000, patience=50, device=0)  # train the model
        metrics = model.val()  # evaluate model performance on the validation set
      # print(metrics.box.map50)
        path = model.export(format="onnx")  # export the model to ONNX format6
        e = time.time()
        finish = e-s
        message = {
                    "message" : f"[집 컴퓨터] : 모델 학습이 완료되었습니다 | 소요시간: {finish} | 모델 mAP: {metrics.box.map50}"
                  }
        requests.post(api_url, headers= headers , data = message)
      
    