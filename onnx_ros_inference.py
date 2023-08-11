#!/usr/bin/env python3
import rospy
import onnxruntime
import numpy as np
import cv2, cv_bridge, time
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from utils import *

class onnx_ros_inference():
    def __init__(self):
        rospy.init_node("rknn_ros_inference", anonymous=True)
        self.bridge = cv_bridge.CvBridge()
        rospy.Subscriber("/camera/color/image_raw", Image, self.color_callback)
        rospy.Subscriber("/aligned_depth_to_color/image_raw", Image, self.align_depth2color_callback)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.color_info_callback)
        rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.depth_info_callback)
        self.segmented_image_publisher = rospy.Publisher("/segmented_image", Image, queue_size=10)
        self.input_width = 640
        self.input_height = 480
        model_name = 'yolov8n-seg'
        model_path = "./model"
        ONNX_MODEL = f"{model_path}/{model_name}-{self.input_height}-{self.input_width}.onnx"
        self.CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis','snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.sess = onnxruntime.InferenceSession(ONNX_MODEL)
        self.images = self.sess.get_inputs()[0].name
        self.output0 = self.sess.get_outputs()[0].name
        self.output1 = self.sess.get_outputs()[1].name
        self.img_size = 288
        self.conf_thres = 0.25
        self.iou_thres = 0.45

    def color_info_callback(self, data):
        self.color_K = np.array(data.K).reshape(3,3)
        #self.color_fx = data.k[0]
        #self.color_fy = data.k[4]
        #self.color_u = data.k[2]
        #self.color_v = data.k[5]
        
    def depth_info_callback(self, data):
        self.depth_K = np.array(data.K).reshape(3,3)
        #self.depth_fx = data.k[0]
        #self.depth_fy = data.k[4]
        #self.depth_u = data.K[2]
        #self.depth_v = data.K[5]
        
    def color_callback(self, data):
        self.color_width = data.width
        self.color_height = data.height
        self.color_image = self.bridge.imgmsg_to_cv2(data)
        
    def align_depth2color_callback(self, data):
        self.depth_width = data.width
        self.depth_height = data.height
        self.depth_image = self.bridge.imgmsg_to_cv2(data)
        self.depth_image = np.array(self.depth_image, dtype=np.float32)
        self.depth_array = self.depth_image/1000.0

    def main(self):
        r = rospy.Rate(30)
        while not rospy.is_shutdown():
            rospy.wait_for_message("/camera/color/image_raw",Image)
            rospy.wait_for_message("/aligned_depth_to_color/image_raw",Image)
            image_4c, image_3c = preprocess(self.color_image, self.input_height, self.input_width)
            outputs = self.sess.run([self.output0, self.output1],{self.images: image_4c.astype(np.float32)}) # (1, 3, input height, input width)
            colorlist = gen_color(len(self.CLASSES))
            results = postprocess(outputs, image_4c, image_3c, self.conf_thres, self.iou_thres, classes=len(self.CLASSES)) ##[box,mask,shape]
            results = results[0]              ## batch=1
            boxes, masks, shape = results
            if isinstance(masks, np.ndarray):
                if masks.ndim == 2:
                    masks = np.expand_dims(masks, axis=0).astype(np.float32)
                vis_img = image_3c.copy()
                mask_img = np.zeros_like(image_3c)
                cls_list = []
                center_list = []
                for box, mask in zip(boxes, masks):
                    cls=int(box[-1])
                    cls_list.append(cls)
                    dummy_img = np.zeros_like(image_3c)
                    dummy_img[mask!=0] = colorlist[int(box[-1])] ## cls=int(box[-1])
                    mask_img[mask!=0] = colorlist[int(box[-1])] ## cls=int(box[-1])
                    centroid = np.mean(np.argwhere(dummy_img),axis=0)
                    if np.isnan(centroid).all() == False:
                        centroid_x, centroid_y = int(centroid[1]), int(centroid[0])
                        center_list.append([centroid_x, centroid_y])
                vis_img = cv2.addWeighted(vis_img,0.5,mask_img,0.5,0)
                for i, box in enumerate (boxes):
                    cls=int(box[-1])
                    cv2.rectangle(vis_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255),3,4)
                    cv2.putText(vis_img, f"{self.CLASSES[cls]}:{round(box[4],2)}", (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                for j in range (len(center_list)):
                    cv2.circle(vis_img, (center_list[j][0], center_list[j][1]), radius=5, color=(0, 0, 255), thickness=-1)
                for i in range (len(self.CLASSES)):
                    num = cls_list.count(i)
                    if num != 0:
                        print(f"Found {num} {self.CLASSES[i]}")
                cv2.imshow("mask_img", mask_img)
                cv2.imshow("vis_img", vis_img)
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "camera_color_optical_frame"
                self.segmented_image_publisher.publish(self.bridge.cv2_to_imgmsg(vis_img,header=header,encoding="bgr8"))
            else:
                print("No segmentation result")
            cv2.waitKey(10)

if __name__ == '__main__':
    inference = onnx_ros_inference()
    inference.main()
