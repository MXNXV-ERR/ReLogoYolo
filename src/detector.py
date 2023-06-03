from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression
import torch

model = attempt_load('C:/Users/MXNXV-ERR/Desktop/MXNXV/Sem6/miniproject/logo/logoyoloongit/LogoYolo/runs/train/yolo_logo_det2/weights/best.pt',map_location=torch.device('cpu'))
model.eval()

img = 'C:/Users/MXNXV-ERR/Desktop/MXNXV/Sem6/miniproject/logo/logoyoloongit/LogoYolo/data/Sample/test/4434414.jpg'

pred = model(img)
pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)
print(pred)

# model = YOLO('runs/train/yolo_logo_det2/weights/best.pt')
# model.predict()