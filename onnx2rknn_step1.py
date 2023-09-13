import os, glob, shutil
from rknn.api import RKNN

input_width = 640
input_height = 480
model_path = "./model"
dataset_path = "./dataset"
config_path = "./config"
dataset_file = "./dataset.txt"
model_name = 'yolov8n-seg'
platform = "rk3588"
ONNX_MODEL = f'{model_path}/{model_name}-{input_height}-{input_width}.onnx'
OUT_NODE = ["output0","output1"]

def get_dataset_txt(dataset_path, dataset_savefile):
    file_data = glob.glob(os.path.join(dataset_path,"*.jpg"))
    with open(dataset_savefile, "w") as f:
        for file in file_data:
            f.writelines(f"{file}\n")

def move_onnx_config():
    file_data = glob.glob("*.onnx")
    for file in file_data:
        shutil.move(file, f"{config_path}/{file}")

if __name__ == '__main__':
    isExist = os.path.exists(dataset_path)
    if not isExist:
        os.makedirs(dataset_path)
        
    isExist = os.path.exists(config_path)
    if not isExist:
        os.makedirs(config_path)

    # Prepare the dataset text file
    get_dataset_txt(dataset_path, dataset_file)

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL, outputs=OUT_NODE)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> hybrid_quantization_step1')
    ret = rknn.hybrid_quantization_step1(dataset=dataset_file, proposal=False)
    if ret != 0:
        print('hybrid_quantization_step1 failed!')
        exit(ret)
    print('done')

    rknn.release()

    print('--> Move hybrid quatization config into config folder')
    shutil.move(f"{model_name}-{input_height}-{input_width}.data", f"{config_path}/{model_name}-{input_height}-{input_width}.data")
    shutil.move(f"{model_name}-{input_height}-{input_width}.model", f"{config_path}/{model_name}-{input_height}-{input_width}.model")
    shutil.move(f"{model_name}-{input_height}-{input_width}.quantization.cfg", f"{config_path}/{model_name}-{input_height}-{input_width}.quantization.cfg")

    print('--> Move onnx config into config folder')
    move_onnx_config()
