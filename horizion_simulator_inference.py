import os, cv2, time, numpy as np
from utils import *
from horizon_tc_ui import HB_ONNXRuntime

conf_thres = 0.25
iou_thres = 0.45
input_width = 640
input_height = 480
input_offset = 128
result_path = "./result"
image_path = "./dataset/bus.jpg"
model_name = 'yolov8n-seg'
model_path = "./model_output"
HORIZON_MODEL = f"{model_path}/{model_name}-{input_height}-{input_width}_quantized_model.onnx"
video_path = "test.mp4"
video_inference = False
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis','snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

if __name__ == '__main__':
    isExist = os.path.exists(result_path)
    if not isExist:
        os.makedirs(result_path)

    sess = HB_ONNXRuntime(HORIZON_MODEL)
    input_name = sess.input_names[0]
    output_name = sess.output_names

    if video_inference == True:
        cap = cv2.VideoCapture(video_path)
        while(True):
            ret, image_3c = cap.read()
            if not ret:
                break
            print('--> Running model for video inference')
            _, image_3c = preprocess(image_3c, input_height, input_width)
            image_4c = np.array(image_3c) / 255.0
            image_4c = np.expand_dims(image_3c, axis=0).astype(np.float32)
            outputs = sess.run(output_name, {input_name: image_4c}, input_offset=input_offset)
            colorlist = gen_color(len(CLASSES)) 
            image_4c = np.transpose(image_4c, (0, 3, 1, 2))  # Channel first
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
        image_4c = np.array(image_3c) / 255.0
        image_4c = np.expand_dims(image_3c, axis=0).astype(np.float32)
        outputs = sess.run(output_name, {input_name: image_4c}, input_offset=input_offset)
        colorlist = gen_color(len(CLASSES)) 
        image_4c = np.transpose(image_4c, (0, 3, 1, 2))  # Channel first
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
