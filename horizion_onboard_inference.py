import os, cv2, time, numpy as np
from utils import *
from hobot_dnn import pyeasy_dnn

conf_thres = 0.25
iou_thres = 0.45
input_width = 640
input_height = 480
input_offset = 128
result_path = "./result"
image_path = "./dataset/bus.jpg"
model_name = 'yolov8n-seg'
model_path = "./model_output"
HORIZON_MODEL = f"{model_path}/{model_name}-{input_height}-{input_width}.bin"
video_path = "test.mp4"
video_inference = False
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis','snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

if __name__ == '__main__':
    isExist = os.path.exists(result_path)
    if not isExist:
        os.makedirs(result_path)

    models = pyeasy_dnn.load(HORIZON_MODEL)
    if video_inference == True:
        cap = cv2.VideoCapture(video_path)
        while(True):
            ret, image_3c = cap.read()
            if not ret:
                break
            print('--> Running model for video inference')
            _, image_3c = preprocess(image_3c, input_height, input_width)
            image_4c = np.transpose(image_3c, (2, 0, 1))  # Channel first
            image_4c = np.expand_dims(image_4c, axis=0).astype(np.uint8)
            start = time.time()
            outputs = models[0].forward(image_3c)
            stop = time.time()
            fps = round(1/(stop-start), 2)
            output0 = outputs[0].buffer
            output1 = outputs[1].buffer
            output0 = np.squeeze(output0)
            output0 = np.expand_dims(output0, axis=0)
            outputs = [output0, output1]
            colorlist = gen_color(len(CLASSES)) 
            results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES)) ##[box,mask,shape]
            results = results[0]              ## batch=1
            boxes, masks, shape = results
            if isinstance(masks, np.ndarray):
                mask_img, vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
                cv2.imshow("mask_img", mask_img)
                cv2.imshow("vis_img", vis_img)
            else:
                print("No segmentation result")
            cv2.waitKey(10)
    else:
        image_3c = cv2.imread(image_path)
        _, image_3c = preprocess(image_3c, input_height, input_width)
        image_4c = np.transpose(image_3c, (2, 0, 1))  # Channel first
        image_4c = np.expand_dims(image_4c, axis=0).astype(np.uint8)
        start = time.time()
        outputs = models[0].forward(image_3c)
        stop = time.time()
        fps = round(1/(stop-start), 2)
        output0 = outputs[0].buffer
        output1 = outputs[1].buffer
        output0 = np.squeeze(output0)
        output0 = np.expand_dims(output0, axis=0)
        outputs = [output0, output1]
        colorlist = gen_color(len(CLASSES)) 
        results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES)) ##[box,mask,shape]
        results = results[0]              ## batch=1
        boxes, masks, shape = results
        if isinstance(masks, np.ndarray):
            mask_img, vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
            print('--> Save inference result')
        else:
            print("No segmentation result")
    print("Horizon inference finish")
    cv2.destroyAllWindows()