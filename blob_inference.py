#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time
from utils import *

conf_thres = 0.25
iou_thres = 0.45
input_width = 640
input_height = 480
model_name = 'yolov8n-seg'
model_path = "./model"
result_path = "./result"
BLOB_MODEL = f'{model_path}/{model_name}-{input_height}-{input_width}.blob'
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis','snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def oakd_setting():
    # Start defining a pipeline
    pipeline = dai.Pipeline()

    pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

    # Define a neural network that will make predictions based on the source frames
    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    detection_nn.setBlobPath(BLOB_MODEL)

    detection_nn.setNumPoolFrames(4)
    detection_nn.input.setBlocking(False)
    detection_nn.setNumInferenceThreads(2)

    # Define a source - color camera
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(input_width,input_height)
    cam.setInterleaved(False)
    cam.preview.link(detection_nn.input)
    cam.setFps(40)

    # Create outputs
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("nn_input")
    xout_rgb.input.setBlocking(False)

    detection_nn.passthrough.link(xout_rgb.input)

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    xout_nn.input.setBlocking(False)

    detection_nn.out.link(xout_nn.input)

    return pipeline

if __name__ == '__main__':
    pipeline = oakd_setting()
    # Pipeline defined, now the device is assigned and pipeline is started
    with dai.Device() as device:
        cams = device.getConnectedCameras()
        device.startPipeline(pipeline)

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        while True:
            # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
            image_3c = q_nn_input.get().getCvFrame()
            image_4c, image_3c = preprocess(image_3c, input_height, input_width)
            layers_name = q_nn.get().getAllLayerNames() # ['output0', 'output1']
            output0 = np.array(q_nn.get().getLayerFp16('output0')).reshape(1, len(CLASSES)+4+32, 6300) # batch size x (number of class)+4+mask number x 6300
            output1 = np.array(q_nn.get().getLayerFp16('output1')).reshape(1, 32, 120, 160) # batch size x mask number x 120 x 160
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
        cv2.destroyAllWindows()
