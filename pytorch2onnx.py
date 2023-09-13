import os, shutil
from ultralytics import YOLO

model_name = 'yolov8n-seg'
input_width = 640
input_height = 480
model_path = "./model"

isExist = os.path.exists(model_path)
if not isExist:
   os.makedirs(model_path)

model = YOLO(f"{model_name}.pt") 
model.export(format="onnx", imgsz=[input_height,input_width], opset=12)
os.rename(f"{model_name}.onnx", f"{model_name}-{input_height}-{input_width}.onnx")
shutil.move(f"{model_name}-{input_height}-{input_width}.onnx", f"./{model_path}/{model_name}-{input_height}-{input_width}.onnx")
shutil.move(f"{model_name}.pt", f"{model_path}/{model_name}.pt")
