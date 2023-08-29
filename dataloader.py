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

from dataset import SingleImageDataset, ImageNetDataset, COCODataset
from dataset import WiderFaceDataset, VOCDataset, CifarDataset, CityscapesDataset


def DataLoader(dataset, transformers, batch_size=1):
    dataset = dataset.transform(transformers)
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset


def SingleImageDataLoader(transformers, image_path, imread_mode='opencv'):
    dataset = SingleImageDataset(image_path, imread_mode)
    loader = DataLoader(dataset, transformers=transformers, batch_size=1)
    return next(loader)


def SingleImageDataLoaderWithOrigin(transformers,
                                    image_path,
                                    imread_mode='opencv'):
    origin_image_dataset = SingleImageDataset(image_path, imread_mode)
    origin_image_loader = DataLoader(origin_image_dataset,
                                     transformers=[],
                                     batch_size=1)

    process_image_dataset = SingleImageDataset(image_path, imread_mode)
    process_image_loader = DataLoader(process_image_dataset,
                                      transformers=transformers,
                                      batch_size=1)
    return [next(origin_image_loader), next(process_image_loader)]


def ImageNetDataLoader(transformers,
                       image_path,
                       label_path=None,
                       imread_mode='opencv',
                       batch_size=None,
                       return_img_name=False):
    dataset = ImageNetDataset(image_path,
                              label_path,
                              imread_mode,
                              return_img_name=return_img_name)
    return DataLoader(dataset,
                      transformers=transformers,
                      batch_size=batch_size)


def COCODataLoader(transformers,
                   image_path,
                   annotations_path=None,
                   batch_size=1,
                   imread_mode='opencv'):
    dataset = COCODataset(image_path, annotations_path, imread_mode)
    return DataLoader(dataset,
                      transformers=transformers,
                      batch_size=batch_size)


def VOCDataLoader(transformers,
                  image_path=None,
                  dataset_path=None,
                  val_txt_path=None,
                  batch_size=1,
                  imread_mode='opencv',
                  segmentation=False):
    dataset = VOCDataset(image_path, dataset_path, val_txt_path, imread_mode,
                         segmentation)
    return DataLoader(dataset,
                      transformers=transformers,
                      batch_size=batch_size)


def WiderFaceDataLoader(transformers,
                        image_path,
                        val_txt_path=None,
                        batch_size=1,
                        imread_mode='opencv'):
    dataset = WiderFaceDataset(image_path, val_txt_path, imread_mode)
    return DataLoader(dataset,
                      transformers=transformers,
                      batch_size=batch_size)


def CifarDataLoader(transformers,
                    image_path,
                    include_label=False,
                    max_len=0,
                    batch_size=1,
                    return_img_name=False):
    dataset = CifarDataset(image_path,
                           include_label=include_label,
                           max_len=max_len,
                           return_img_name=return_img_name)
    return DataLoader(dataset,
                      transformers=transformers,
                      batch_size=batch_size)


def CityscapesDataLoader(transformers,
                         imageset_path,
                         val_path=None,
                         batch_size=1,
                         imread_mode='opencv',
                         return_img_name=False):
    loader = CityscapesDataset(imageset_path, val_path, imread_mode,
                               return_img_name)
    return DataLoader(loader, transformers=transformers, batch_size=batch_size)
