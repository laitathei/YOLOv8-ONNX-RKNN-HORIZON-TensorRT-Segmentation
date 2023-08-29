# Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import cv2
import pickle
import warnings
import skimage.io
import numpy as np
from PIL import Image
from collections import defaultdict
from dataset_consts import COCO_CLASSES, VOC_CLASSES

warnings.filterwarnings("ignore", "(Possibly )?Corrupt EXIF data", UserWarning)


class Dataset(object):
    def __init__(self):
        return

    def __iter__(self):
        return self

    def __next__(self):
        return self.perform()

    def perform(self):
        raise AssertionError("Shouldn't reach here.")

    def batch(self, batch_size):
        return _BatchDataset(self, batch_size)

    def transform(self, transformers):
        return _TransformDataset(self, transformers)


class SingleImageDataset(Dataset):
    """
    A basic image dataset, it will imread one single image for model inference
    """
    def __init__(self, image_path, imread_mode='opencv'):
        super(SingleImageDataset, self).__init__()
        self.image_read_mode = imread_mode
        self.image_read_method = _get_image_read_method(imread_mode)
        self.image_path = image_path

    def perform(self):
        if self.image_read_mode == 'skimage':
            image = self.image_read_method(self.image_path).astype(np.float32)
        elif self.image_read_mode == 'opencv':
            image = self.image_read_method(self.image_path).astype(np.uint8)
        else:
            raise ValueError(
                f'invalid image read mode: {self.image_read_mode}')
        if image.ndim != 3:  # expend gray scale image to three channels
            image = image[..., np.newaxis]
            image = np.concatenate([image, image, image], axis=-1)
        return [image]


class ImageNetDataset(Dataset):
    """
    ImageNet validation dataset
    """
    def __init__(self,
                 image_path,
                 label_path=None,
                 imread_mode='opencv',
                 return_img_name=False):
        super(ImageNetDataset, self).__init__()
        self.image_read_mode = imread_mode
        self.image_read_method = _get_image_read_method(imread_mode)
        if label_path:
            file_list, img2label = self._build_im2label(image_path, label_path)
            self._gen = self._generator(file_list, return_img_name, img2label)
        else:
            file_list = self._get_image_list(image_path)
            self._gen = self._generator(file_list)

    def perform(self):
        return next(self._gen)

    def _get_image_list(self, image_path):
        image_name_list = []
        image_file_list = sorted(os.listdir(image_path))
        for image in image_file_list:
            image_name_list.append(os.path.join(image_path, image))
        return image_name_list

    def _build_im2label(self, image_path, label_path):
        img2label = dict()
        image_name_list = []
        with open(label_path) as file:
            line = file.readline()
            while line:
                img, label = line[:-1].split(" ")
                one_image = os.path.join(image_path, img)
                img2label[one_image] = int(label)
                image_name_list.append(one_image)
                line = file.readline()
        return image_name_list, img2label

    def _generator(self, file_list, return_img_name=False, img2label=None):
        for idx, image_path in enumerate(file_list):
            if self.image_read_mode == 'skimage':
                image = self.image_read_method(image_path).astype(np.float32)
            elif self.image_read_mode == 'opencv':
                image = self.image_read_method(image_path).astype(np.uint8)
            else:
                raise ValueError(
                    f'invalid image read mode: {self.image_read_mode}')
            if image.ndim != 3:  # expend gray scale image to three channels
                image = image[..., np.newaxis]
                image = np.concatenate([image, image, image], axis=-1)
            if img2label:
                label = img2label[image_path]
                if return_img_name:
                    yield [image, label, os.path.basename(image_path)]
                else:
                    yield [image, label]
            else:
                yield [image]


class COCODataset(Dataset):
    """
    coco validation dataset
    """
    def __init__(self, image_path, annotations_path, imread_mode='opencv'):
        from pycocotools.coco import COCO
        super(COCODataset, self).__init__()
        self.image_read_mode = imread_mode
        self.image_read_method = _get_image_read_method(imread_mode)

        if annotations_path:
            self.annotations_path = annotations_path
            self.image_path = image_path
            self.classes = COCO_CLASSES
            self.coco = COCO(self.annotations_path)
            self.image_ids = sorted(self.coco.getImgIds())
            class_cat = self.coco.dataset["categories"]
            self.id2name = {}
            for (i, cat) in enumerate(class_cat):
                self.id2name[cat['id']] = cat['name']

            self._gen = self._generator()
        else:
            self.image_path = image_path
            self._gen = self._generator_without_anno()

    def _generator_without_anno(self):
        """calibration data generator without annotation"""
        file_name_dir = sorted(os.listdir(self.image_path))
        for file in file_name_dir:
            image_path = os.path.join(self.image_path, file)
            if self.image_read_mode == 'skimage':
                image = self.image_read_method(image_path).astype(np.float32)
            elif self.image_read_mode == 'opencv':
                image = self.image_read_method(image_path).astype(np.uint8)
            else:
                raise ValueError(
                    f'invalid image read mode: {self.image_read_mode}')
            yield [image]

    def _generator(self):
        for entry in self.coco.loadImgs(self.image_ids):
            filename = entry['file_name']
            if self.image_read_mode == 'skimage':
                image = self.image_read_method(
                    os.path.join(self.image_path, filename)).astype(np.float32)
            elif self.image_read_mode == 'opencv':
                image = self.image_read_method(
                    os.path.join(self.image_path, filename)).astype(np.uint8)
            else:
                raise ValueError(
                    f'invalid image read mode: {self.image_read_mode}')
            org_height, org_width, _ = image.shape

            ann_ids = self.coco.getAnnIds(imgIds=entry['id'])
            annotations = self.coco.loadAnns(ann_ids)

            height = entry['height']
            width = entry['width']

            info_dict = {
                'origin_shape': (org_height, org_width),
                'image_name': filename,
                'class_name': [],
                'class_id': [],
                'bbox': []
            }
            if len(annotations) > 0:
                for ann in annotations:
                    x1, y1, w, h = ann['bbox']
                    x2 = x1 + w
                    y2 = y1 + h
                    x1 = np.minimum(width, np.maximum(0, x1))
                    y1 = np.minimum(height, np.maximum(0, y1))
                    x2 = np.minimum(width, np.maximum(0, x2))
                    y2 = np.minimum(height, np.maximum(0, y2))
                    cat_name = self.id2name[ann['category_id']]
                    class_id = self.classes.index(cat_name)
                    info_dict['class_name'].append(cat_name)
                    info_dict['class_id'].append(class_id)
                    info_dict['bbox'].append([x1, y1, x2, y2])
            yield [image, info_dict]

    def perform(self):
        return next(self._gen)


class VOCDataset(Dataset):
    """
    A voc validation dataset
    """
    def __init__(self,
                 image_path,
                 dataset_path,
                 val_txt_path,
                 imread_mode='opencv',
                 segmentation=False):

        super(VOCDataset, self).__init__()
        self.image_read_mode = imread_mode
        self.image_read_method = _get_image_read_method(imread_mode)

        self.segmentation = segmentation
        if dataset_path and val_txt_path:
            if self.segmentation is True:
                self.seg_path = os.path.join(dataset_path, "SegmentationClass")
            self.annotations_path = os.path.join(dataset_path, "Annotations")
            self.image_path = os.path.join(dataset_path, "JPEGImages")
            self.val_txt_path = val_txt_path
            self.classes = VOC_CLASSES
            self._gen = self._generator()
        elif image_path is not None:
            self.image_path = image_path
            self._gen = self._generator_without_anno()
        else:
            raise ValueError(
                "imageset_path or (dataset_path and val_txt_path) is not set ")

    def _generator_without_anno(self):
        """calibration data generator without annotation"""
        file_name_dir = os.listdir(self.image_path)
        for file in file_name_dir:
            image_path = os.path.join(self.image_path, file)
            if self.image_read_mode == 'skimage':
                image = self.image_read_method(image_path).astype(np.float32)
            elif self.image_read_mode == 'opencv':
                image = self.image_read_method(image_path).astype(np.uint8)
            else:
                raise ValueError(
                    f'invalid image read mode: {self.image_read_mode}')
            yield [image]

    def _generator(self):
        import xml.etree.ElementTree as ET
        val_file = open(self.val_txt_path, 'r')
        for f in val_file:
            file_name = f.strip() + '.xml'
            annotation_path = os.path.join(self.annotations_path, file_name)
            tree = ET.ElementTree(file=annotation_path)
            root = tree.getroot()
            object_set = root.findall('object')
            image_path = root.find('filename').text
            if self.image_read_mode == 'skimage':
                image = self.image_read_method(
                    os.path.join(self.image_path,
                                 image_path)).astype(np.float32)
            elif self.image_read_mode == 'opencv':
                image = self.image_read_method(
                    os.path.join(self.image_path, image_path)).astype(np.uint8)
            else:
                raise ValueError(
                    f'invalid image read mode: {self.image_read_mode}')
            org_h, org_w, _ = image.shape
            info_dict = {}

            info_dict['origin_shape'] = (org_h, org_w)
            info_dict['image_name'] = image_path

            if self.segmentation is True:
                seg_file = f.strip() + '.png'
                seg_file = os.path.join(self.seg_path, seg_file)
                seg = Image.open(seg_file)
                seg = np.array(seg)
                seg[seg > 20] = 0
                info_dict['seg'] = seg
            else:
                info_dict['class_name'] = []
                info_dict['class_id'] = []
                info_dict['bbox'] = []
                info_dict["difficult"] = []
                for obj in object_set:
                    obj_name = obj.find('name').text
                    bbox = obj.find('bndbox')
                    x1 = int(bbox.find('xmin').text)
                    y1 = int(bbox.find('ymin').text)
                    x2 = int(bbox.find('xmax').text)
                    y2 = int(bbox.find('ymax').text)
                    difficult = int(obj.find("difficult").text)
                    bbox_loc = [x1, y1, x2, y2]

                    info_dict['class_name'].append(obj_name)
                    info_dict['class_id'].append(self.classes[obj_name][0])
                    info_dict['bbox'].append(bbox_loc)
                    info_dict["difficult"].append(difficult)

            yield [image, info_dict]

    def perform(self):
        return next(self._gen)


class WiderFaceDataset(Dataset):
    """
    wider face dataset
    """
    def __init__(self, image_path, val_txt_path, imread_mode='opencv'):

        super(WiderFaceDataset, self).__init__()
        self.image_read_mode = imread_mode
        self.image_read_method = _get_image_read_method(imread_mode)

        if val_txt_path:
            self.image_path = image_path
            self.val_txt_path = val_txt_path
            self._gen = self._generator()
        else:
            self.image_path = image_path
            self._gen = self._generator_without_anno()

    def _generator_without_anno(self):
        """calibration data generator without annotation"""
        file_name_dir = os.listdir(self.image_path)
        for file in file_name_dir:
            image_path = os.path.join(self.image_path, file)
            if self.image_read_mode == 'skimage':
                image = self.image_read_method(image_path).astype(np.float32)
            elif self.image_read_mode == 'opencv':
                image = self.image_read_method(image_path).astype(np.uint8)
            else:
                raise ValueError(
                    f'invalid image read mode: {self.image_read_mode}')
            yield [image]

    def _generator(self):
        with open(self.val_txt_path, 'r') as val_file:
            content = [line.strip() for line in val_file]
            index = 0
            while index < len(content):
                image_name = content[index]
                if self.image_read_mode == 'skimage':
                    image = self.image_read_method(
                        os.path.join(self.image_path,
                                     image_name)).astype(np.float32)
                elif self.image_read_mode == 'opencv':
                    image = self.image_read_method(
                        os.path.join(self.image_path,
                                     image_name)).astype(np.uint8)
                else:
                    raise ValueError(
                        f'invalid image read mode: {self.image_read_mode}')
                org_h, org_w, _ = image.shape
                info_dict = {}
                info_dict['origin_shape'] = (org_h, org_w)
                info_dict['image_name'] = image_name
                info_dict['bbox'] = []
                info_dict['blur'] = []
                info_dict['expression'] = []
                info_dict['illumination'] = []
                info_dict['invalid'] = []
                info_dict['occlusion'] = []
                info_dict['pose'] = []
                num_bbox = int(content[index + 1])
                index += 2
                for box in range(num_bbox):
                    box_info = [
                        int(i) for i in content[index + box].split(" ")
                    ]
                    assert len(
                        box_info
                    ) == 10, "invalid box info, make sure val.txt is unbroken"
                    x_min = int(box_info[0])
                    y_min = int(box_info[1])
                    x_max = int(box_info[2])
                    y_max = int(box_info[3])
                    box_loc = [x_min, y_min, x_max, y_max]
                    info_dict['bbox'].append(box_loc)
                    info_dict['blur'].append(int(box_info[4]))
                    info_dict['expression'].append(int(box_info[5]))
                    info_dict['illumination'].append(int(box_info[6]))
                    info_dict['invalid'].append(int(box_info[7]))
                    info_dict['occlusion'].append(int(box_info[8]))
                    info_dict['pose'].append(int(box_info[9]))

                index += num_bbox
                yield [image, info_dict]

    def perform(self):
        return next(self._gen)


class CifarDataset(Dataset):
    """
    cifar 10 dataset
    """
    def __init__(self,
                 cifar_path,
                 include_label=True,
                 max_len=0,
                 return_img_name=False):
        super(CifarDataset, self).__init__()
        self.data = []
        self.targets = []
        self.include_label = include_label
        self.max_len = max_len
        self.return_img_name = return_img_name

        for fs in os.listdir(cifar_path):
            if fs == "test_batch":
                fs = os.path.join(cifar_path, fs)
                with open(fs, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    if 'labels' in entry:
                        self.targets.extend(entry['labels'])
                    else:
                        self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32,
                                                 32).astype(np.float32)
        self._gen = self._generator()

    def _generator(self):
        count = 0
        for i in range(len(self.data)):
            count += 1
            if self.max_len != 0 and count > self.max_len:
                raise StopIteration
            if self.include_label is True:
                if self.return_img_name:
                    yield [self.data[i], self.targets[i], ""]
                else:
                    yield [self.data[i], self.targets[i]]
            else:
                yield [self.data[i]]

    def __len__(self):
        return len(self.data)

    def perform(self):
        return next(self._gen)


class CityscapesDataset(Dataset):
    """
    A generator for cityspace dataset
    """
    pixLabels = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 1,
        8: 2,
        9: 0,
        10: 0,
        11: 3,
        12: 4,
        13: 5,
        14: 0,
        15: 0,
        16: 0,
        17: 6,
        18: 0,
        19: 7,
        20: 8,
        21: 9,
        22: 10,
        23: 11,
        24: 12,
        25: 13,
        26: 14,
        27: 15,
        28: 16,
        29: 0,
        30: 0,
        31: 17,
        32: 18,
        33: 19,
        -1: 0
    }

    def __init__(self,
                 imageset_path,
                 val_path=None,
                 imread_mode='opencv',
                 return_img_name=False):
        super(CityscapesDataset, self).__init__()
        self.image_read_mode = imread_mode
        if imread_mode == 'opencv':
            self.image_read_method = cv2.imread
        elif imread_mode == 'skimage' or imread_mode == 'caffe':
            self.image_read_method = lambda x: skimage.img_as_float(
                skimage.io.imread(x)).astype(np.float32)
        else:
            raise ValueError(
                "Unsupport image read method:{}".format(imread_mode))

        self.imageset_path = imageset_path
        self.val_path = val_path
        if self.val_path:
            self._gen = self._generator(return_img_name)
        else:
            self._gen = self._generator_without_anno()

    def _generator_without_anno(self):
        """calibration data generator without annotation"""
        image_path_list = []

        def gen_dir(one_path):
            """load all image_path recursively"""
            file_list = os.listdir(one_path)
            for file in file_list:
                curr_path = os.path.join(one_path, file)
                if os.path.isdir(curr_path):
                    gen_dir(one_path)
                else:
                    image_path_list.append(curr_path)

        gen_dir(self.imageset_path)
        for image_path in image_path_list:
            if self.image_read_mode == 'skimage':
                image = self.image_read_method(image_path).astype(np.float32)
            elif self.image_read_mode == 'opencv':
                image = self.image_read_method(image_path).astype(np.uint8)
            else:
                raise ValueError(
                    f'invalid image read mode: {self.image_read_mode}')
            yield [image]

    def _generator(self, return_img_name=False):
        def gen_dir(one_path, image_list):
            """load all image_path recursively"""
            file_list = os.listdir(one_path)
            for file in file_list:
                curr_path = os.path.join(one_path, file)
                if os.path.isdir(curr_path):
                    gen_dir(curr_path, image_list)
                else:
                    image_list.append(curr_path)

        gt_path_list, image_path_list = [], []
        gen_dir(self.imageset_path, image_path_list)
        gen_dir(self.val_path, gt_path_list)

        gt_path_list = [
            png_gt for png_gt in gt_path_list
            if png_gt.endswith("_labelIds.png")
        ]
        assert len(image_path_list) == len(gt_path_list), \
            "the number of image:{} is not equal to the number of label:{}" \
                .format(len(image_path_list), len(gt_path_list))

        image_path_list = sorted(image_path_list)
        gt_path_list = sorted(gt_path_list)
        for image_path, gt_path in zip(image_path_list, gt_path_list):
            if self.image_read_mode == 'skimage':
                image = self.image_read_method(image_path).astype(np.float32)
            elif self.image_read_mode == 'opencv':
                image = self.image_read_method(image_path).astype(np.uint8)
            else:
                raise ValueError(
                    f'invalid image read mode: {self.image_read_mode}')
            gt = cv2.imread(gt_path, 0)
            binary_gt = np.zeros_like(gt).astype(np.int32)
            for key in self.pixLabels.keys():
                index = np.where(gt == key)
                binary_gt[index] = self.pixLabels[key] - 1
            if return_img_name:
                yield [image, binary_gt, image_path]
            else:
                yield [image, binary_gt]

    def perform(self):
        return next(self._gen)


class _TransformDataset(Dataset):
    def __init__(self, dataset, transformers):
        super(_TransformDataset, self).__init__()
        self._dataset = dataset
        self._trans = transformers

    def perform(self):
        data = self._dataset.perform()
        for tran in self._trans:
            data[0] = tran([data[0]])[0]
        return data


class _BatchDataset(Dataset):
    def __init__(self, dataset, batch_size):
        super(_BatchDataset, self).__init__()
        self._dataset = dataset
        self._batch_size = batch_size

    def perform(self):
        batch_map = defaultdict(list)
        for _ in range(self._batch_size):
            try:
                data = self._dataset.perform()
                for i in range(len(data)):
                    batch_map['data%d' % (i)].append(data[i])
            except StopIteration:
                if len(batch_map) > 0:
                    break
                else:
                    raise StopIteration
        data = list(batch_map.values())
        for i in range(len(data)):
            data[i] = np.array(data[i])
        if len(data) == 1:
            return data[0]
        return data


def _get_image_read_method(mode):
    if mode == 'opencv':
        return lambda x: cv2.imread(x)
    elif mode == 'skimage' or mode == 'caffe':
        return lambda x: skimage.img_as_float(skimage.io.imread(x)).astype(
            np.float32)
    else:
        raise ValueError("Unsupport read method:{}".format(mode))
