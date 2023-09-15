import os, time, json
from datetime import datetime
import tensorrt as trt

task = "segment"
batch_size = 1
input_width = 640
input_height = 480
precision = "FP16" # FP16, INT8
workspace = 1 # number of GB
model_path = "./model"
model_name = 'yolov8n-seg'
ONNX_MODEL = f'{model_path}/{model_name}-{input_height}-{input_width}.onnx'
TensorRT_MODEL = f'{model_path}/{model_name}-{input_height}-{input_width}.engine'
CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

if __name__ == '__main__':
    # Build TensorRT engine from Onnx file
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30 # in terms of GB

    #config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (2 ** 30)) # in terms of GB
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    # Parse model file
    if not os.path.exists(ONNX_MODEL):
        print('ONNX file {} not found, please run pytorch2onnx.py first to generate it.'.format(ONNX_MODEL))
        exit(0)
    if not parser.parse_from_file(ONNX_MODEL):
        raise RuntimeError(f'failed to load ONNX file: {ONNX_MODEL}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    print("Network Description")
    for input in inputs:
        print("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
    for output in outputs:
        print("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))

    if builder.platform_has_fast_fp16 and precision == "FP16": # build with fp16
        config.set_flag(trt.BuilderFlag.FP16)
    elif builder.platform_has_fast_int8 and precision == "INT8": # build with int8
        config.set_flag(trt.BuilderFlag.INT8)

    start = time.time()
    metadata = {
        'description': "Ultralytics YOLOv8n-seg model (untrained)",
        'author': 'Ultralytics',
        'license': 'MIT License',
        'date': datetime.now().isoformat(),
        'version': "8.0.147",
        'stride': 32,
        'task': task,
        'batch': batch_size,
        'imgsz': [input_height, input_width],
        'names': CLASSES}  # model metadata

    # Write file
    with builder.build_engine(network, config) as engine, open(TensorRT_MODEL, 'wb') as t:
        # Metadata
        meta = json.dumps(metadata)
        t.write(len(meta).to_bytes(4, byteorder='little', signed=True))
        t.write(meta.encode())
        # Model
        t.write(engine.serialize())
    print("used time: ", time.time()-start)
    print("Saved TensorRT Engine")
    
