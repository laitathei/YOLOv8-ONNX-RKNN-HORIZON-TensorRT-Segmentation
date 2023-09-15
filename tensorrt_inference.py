import cv2, json
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from utils import *
import torch
from collections import OrderedDict, namedtuple

conf_thres = 0.25
iou_thres = 0.45
input_width = 640
input_height = 480
result_path = "./result"
image_path = "./dataset/bus.jpg"
model_name = 'yolov8n-seg'
model_path = "./model"
ONNX_MODEL = f'{model_path}/{model_name}-{input_height}-{input_width}.onnx'
TensorRT_MODEL = f'{model_path}/{model_name}-{input_height}-{input_width}.engine'
video_path = "test.mp4"
video_inference = False
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis','snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def main():
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    device = torch.device('cuda:0')
    # Read file
    with open(TensorRT_MODEL, 'rb') as f, trt.Runtime(logger) as runtime:
        meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
        metadata = json.loads(f.read(meta_len).decode('utf-8'))  # read metadata
        model = runtime.deserialize_cuda_engine(f.read())  # read engine
    context = model.create_execution_context()
    bindings = OrderedDict()
    input_names = []
    output_names = []
    for i in range(model.num_bindings):
        name = model.get_binding_name(i) 
        dtype = trt.nptype(model.get_binding_dtype(i))
        if model.binding_is_input(i):
            input_names.append(name)
        else:  # output
            output_names.append(name)
        shape = tuple(context.get_binding_shape(i))
        im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
        bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())

    if video_inference == True:    
        cap = cv2.VideoCapture(video_path)
        while(True):
            ret, image_3c = cap.read()
            if not ret:
                break
            image_4c, image_3c = preprocess(image_3c, input_height, input_width)
            image_4c = image_4c.astype(np.float32)
            image_4c = torch.from_numpy(image_4c).to(device)
            if image_4c.dtype != torch.float16:
                image_4c = image_4c.half()  # to FP16
            binding_addrs['images'] = int(image_4c.data_ptr())
            context.execute_v2(list(binding_addrs.values()))
            outputs = [bindings[x].data.cpu().numpy() for x in sorted(output_names)] # put the result from gpu to cpu and convert to numpy
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
        image_4c, image_3c = preprocess(image_3c, input_height, input_width)
        image_4c = image_4c.astype(np.float32)
        image_4c = torch.from_numpy(image_4c).to(device)
        if image_4c.dtype != torch.float16:
            image_4c = image_4c.half()  # to FP16
        binding_addrs['images'] = int(image_4c.data_ptr())
        context.execute_v2(list(binding_addrs.values()))
        outputs = [bindings[x].data.cpu().numpy() for x in sorted(output_names)] # put the result from gpu to cpu and convert to numpy
        colorlist = gen_color(len(CLASSES)) 
        results = postprocess(outputs, image_4c, image_3c, conf_thres, iou_thres, classes=len(CLASSES)) ##[box,mask,shape]
        results = results[0]              ## batch=1
        boxes, masks, shape = results
        if isinstance(masks, np.ndarray):
            mask_img, vis_img = vis_result(image_3c,  results, colorlist, CLASSES, result_path)
            print('--> Save inference result')
        else:
            print("No segmentation result")
    print("TensorRT inference finish")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
