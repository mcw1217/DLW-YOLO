import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO('./runs/detect/train57/weights/best.pt')

results = model('./dataset/neu-det/val/images', conf=0.5, save=True,line_thickness =3, show_conf=False, show_labels=False)

# results = model.val()
# print(results[300])

# res_plotted = results[300].plot()
# plt.figure(figsize=(12,12))
# plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
# plt.show()
