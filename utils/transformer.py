# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
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

import cv2
import math
import numpy as np
import numbers
from skimage.transform import resize as sresize
from scipy.ndimage import zoom
from PIL import Image

INTERPOLATION_DICT = {
    'INTER_CUBIC': cv2.INTER_CUBIC,
}


class Transformer(object):
    def __init__(self):
        pass

    def __call__(self, data):
        result = []
        for i in range(len(data)):
            self.pre_process(data, data[i])
            result.append(self.run_transform(data[i]))
            # Adapt all data operations
            self.post_process(result, result[i])
        return result

    def run_transform(self, data):
        return data

    def pre_process(self, data_list, data):
        pass

    def post_process(self, data_list, data):
        pass


class AddTransformer(Transformer):
    def __init__(self, value):
        self.value = value
        super(AddTransformer, self).__init__()

    def run_transform(self, data):
        data = data.astype(np.float32)
        data += self.value
        return data


class MeanTransformer(Transformer):
    def __init__(self, means, data_format="CHW"):
        self.means = means
        self.data_format = data_format
        super(MeanTransformer, self).__init__()

    def run_transform(self, data):
        if self.data_format is "HWC":
            data = data - self.means
        else:
            data = data - self.means[:, np.newaxis, np.newaxis]
        data = data.astype(np.float32)
        return data


class ScaleTransformer(Transformer):
    def __init__(self, scale_value):
        self.scale_value = scale_value
        super(ScaleTransformer, self).__init__()

    def run_transform(self, data):
        data = data * self.scale_value
        data = data.astype(np.float32)
        return data


class NormalizeTransformer(Transformer):
    def __init__(self, std):
        self.std = std
        super(NormalizeTransformer, self).__init__()

    def run_transform(self, data):
        data = data / self.std
        data = data.astype(np.float32)
        return data


class TransposeTransformer(Transformer):
    def __init__(self, order):
        self.order = order
        super(TransposeTransformer, self).__init__()

    def run_transform(self, data):
        data = np.transpose(data, self.order)
        return data


class HWC2CHWTransformer(Transformer):
    def __init__(self):
        self.transformer = TransposeTransformer((2, 0, 1))

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class CHW2HWCTransformer(Transformer):
    def __init__(self):
        self.transformer = TransposeTransformer((1, 2, 0))

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class CenterCropTransformer(Transformer):
    def __init__(self, crop_size, data_type="float"):
        self.crop_size = crop_size
        self.data_type = data_type
        super(CenterCropTransformer, self).__init__()

    def run_transform(self, image):
        resize_height, resize_width, _ = image.shape
        resize_up = resize_height // 2 - self.crop_size // 2
        resize_left = resize_width // 2 - self.crop_size // 2
        image = image[resize_up:resize_up + self.crop_size,
                      resize_left:resize_left + self.crop_size, :]
        if self.data_type == "uint8":
            image = image.astype(np.uint8)
        else:
            image = image.astype(np.float32)
        return image


class PILCenterCropTransformer(Transformer):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        super(PILCenterCropTransformer, self).__init__()

    def run_transform(self, data):
        img = Image.fromarray(data.astype('uint8'), 'RGB')
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        data = np.array(
            img.crop((crop_left, crop_top, crop_left + crop_width,
                      crop_top + crop_height))).astype(np.float32)
        return data


class LongSideCropTransformer(Transformer):
    def __init__(self):
        super(LongSideCropTransformer, self).__init__()

    # mobilenetv1, v2 will register this method
    def run_transform(self, image):
        height, width, _ = image.shape
        if height < width:
            off = (width - height) // 2
            image = image[:, off:off + height]
        else:
            off = (height - width) // 2
            image = image[off:off + height, :]
        image = image.astype(np.float32)
        return image


class PadResizeTransformer(Transformer):
    def __init__(self, target_size, pad_value=127., pad_position='boundary'):
        self.target_size = target_size
        self.pad_value = pad_value
        self.pad_position = pad_position
        super(PadResizeTransformer, self).__init__()

    def run_transform(self, image):
        target_h, target_w = self.target_size
        image_h, image_w, _ = image.shape
        scale = min(target_w * 1.0 / image_w, target_h * 1.0 / image_h)
        new_h, new_w = int(scale * image_h), int(scale * image_w)
        resize_image = cv2.resize(image, (new_w, new_h))
        pad_image = np.full(shape=[target_h, target_w, 3],
                            fill_value=self.pad_value).astype(np.float32)
        if self.pad_position == 'boundary':
            # valid data is at center.
            dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2
            pad_image[dh:new_h + dh, dw:new_w + dw, :] = resize_image
        elif self.pad_position == 'bottom_right':
            # valid data is at top_left.
            pad_image[:new_h, :new_w, :] = resize_image
        else:
            raise ValueError('Unsupported pad position setting: {}'.format(
                self.pad_position))
        image = pad_image
        image = image.astype(np.float32)
        return image


class ResizeTransformer(Transformer):
    # todo resize 是否需要支持多种插值方法（interpretation参数）
    #   The order of interpolation. The order has to be in the range 0-5:
    # 0: Nearest-neighbor
    # 1: Bi-linear (default)
    # 2: Bi-quadratic
    # 3: Bi-cubic
    # 4: Bi-quartic
    # 5: Bi-quintic
    def __init__(self,
                 target_size,
                 mode='skimage',
                 method=1,
                 data_type="float",
                 interpolation=""):
        self.target_size = target_size
        self.resize_mode = mode
        # Resize method for skimage
        self.resize_method = method
        self.data_type = data_type
        if mode == 'opencv' and interpolation != "" and interpolation not in INTERPOLATION_DICT.keys(
        ):
            raise ValueError(
                f" resize method is set to {mode}. It does not support interpolation method {interpolation} at the moment."
            )
        self.interpolation = interpolation
        super(ResizeTransformer, self).__init__()

    def run_transform(self, data):
        if self.resize_mode == 'skimage':
            return self.skimage_resize(data)
        elif self.resize_mode == 'opencv':
            return self.opencv_resize(data)
        else:
            raise ValueError("unsupport resize mode:{}.(" + \
                "skimage and opencv are supported)".format(self.resize_mode))

    def opencv_resize(self, data):
        target_h, target_w = self.target_size
        if self.interpolation and self.interpolation in INTERPOLATION_DICT.keys(
        ):
            data = cv2.resize(
                data, (target_w, target_h),
                interpolation=INTERPOLATION_DICT[self.interpolation])
        else:
            data = cv2.resize(data, (target_w, target_h))

        if self.data_type == "uint8":
            data = data.astype(np.uint8)
        else:
            data = data.astype(np.float32)
        return data

    def skimage_resize(self, image):
        """
        im : (H x W x K)
        interp_order : interpolation order, default is linear.
        """
        if image.shape[-1] == 1 or image.shape[-1] == 3:
            im_min, im_max = image.min(), image.max()
            if im_max > im_min:
                # skimage is fast but only understands {1,3} channel images in [0, 1].
                im_std = (image - im_min) / (im_max - im_min)
                resized_std = sresize(im_std,
                                      self.target_size,
                                      order=self.resize_method
                                      )  # should be interp_order=1 default
                resized_im = resized_std * (im_max - im_min) + im_min
            else:
                # the image is a constant -- avoid divide by 0
                ret = np.empty((self.target_size[0], self.target_size[1],
                                image.shape[-1]),
                               dtype=np.float32)
                ret.fill(im_min)
                return ret
        else:
            # ndimage interpolates anything but more slowly.
            scale = tuple(
                np.array(self.target_size, dtype=float) /
                np.array(image.shape[:2]))
            resized_im = zoom(image, scale + (1, ), order=1)
        if self.data_type == "uint8":
            image = resized_im.astype(np.uint8)
            image = image.astype(np.uint8)
        else:
            image = resized_im.astype(np.float32)
            image = image.astype(np.float32)
        return image


class PILResizeTransformer(Transformer):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        super(PILResizeTransformer, self).__init__()

    def run_transform(self, data):
        img = Image.fromarray(data.astype('uint8'), 'RGB')
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                data = np.array(img)
                return data
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                data = np.array(img.resize((ow, oh), self.interpolation))
                return data
            else:
                oh = self.size
                ow = int(self.size * w / h)
                data = np.array(img.resize((ow, oh), self.interpolation))
                return data
        else:
            data = np.array(img.resize(self.size[::-1], self.interpolation))
            return data


class ShortLongResizeTransformer(Transformer):
    def __init__(self, short_size, long_size, include_im=True):
        self.short_size = short_size
        self.long_size = long_size
        self.include_im = include_im
        self.im_scale = 0.0
        super(ShortLongResizeTransformer, self).__init__()

    def run_transform(self, image):
        height, width, _ = image.shape
        im_size_min = min(height, width)
        im_size_max = max(height, width)

        self.im_scale = float(self.short_size) / float(im_size_min)
        if round(self.im_scale * im_size_max) > self.long_size:
            self.im_scale = float(self.long_size) / float(im_size_max)
        image = cv2.resize(image,
                           None,
                           None,
                           fx=self.im_scale,
                           fy=self.im_scale)
        return image

    def post_process(self, data, image):
        if self.include_im is True:
            im_info = np.array([image.shape[0], image.shape[1], self.im_scale],
                               dtype=np.float32)
            data.append(im_info)


class PadTransformer(Transformer):
    def __init__(self, size_divisor=128, target_size=512):
        self.target_size = target_size
        self.size_divisor = size_divisor
        super(PadTransformer, self).__init__()

    def run_transform(self, image):
        h, w, c = image.shape
        smallest_side = max(h, w)
        scale = self.target_size / smallest_side
        image = cv2.resize(
            image,
            (int(round(w * scale)), int(round((h * scale)))),
            interpolation=cv2.INTER_LINEAR,
        )

        h, w, c = image.shape
        h_padded = math.ceil(h / self.size_divisor) * self.size_divisor
        w_padded = math.ceil(w / self.size_divisor) * self.size_divisor

        new_image = np.zeros((h_padded, w_padded, c), dtype=np.uint8)
        new_image[:h, :w, :] = image.astype(np.uint8)
        image = new_image
        image = image.astype(np.float32)
        return image


class ShortSideResizeTransformer(Transformer):
    def __init__(self, short_size, data_type='float', interpolation=""):
        self.short_size = short_size
        self.data_type = data_type
        if interpolation != "" and interpolation not in INTERPOLATION_DICT.keys(
        ):
            raise ValueError(
                f"The transformer does not supported interpolation method {interpolation} at the moment "
            )
        self.interpolation = interpolation
        super(ShortSideResizeTransformer, self).__init__()

    def run_transform(self, image):
        height, width, _ = image.shape
        if height < width:
            off = width / height
            if self.interpolation and self.interpolation in INTERPOLATION_DICT.keys(
            ):
                image = cv2.resize(
                    image, (int(self.short_size * off), self.short_size),
                    interpolation=INTERPOLATION_DICT[self.interpolation])
            else:
                image = cv2.resize(
                    image, (int(self.short_size * off), self.short_size))
        else:
            off = height / width
            if self.interpolation and self.interpolation in INTERPOLATION_DICT.keys(
            ):
                image = cv2.resize(
                    image, (self.short_size, int(self.short_size * off)),
                    interpolation=INTERPOLATION_DICT[self.interpolation])
            else:
                image = cv2.resize(
                    image, (self.short_size, int(self.short_size * off)))
        if self.data_type == "uint8":
            image = image.astype(np.uint8)
        else:
            image = image.astype(np.float32)
        return image


class PaddedCenterCropTransformer(Transformer):
    # 仅用于 EfficientNet-lite相关实例模型
    # 请参考 EfficientNet 源码的models/official/efficientnet/preprocessing.py
    # 以原图短边的 ( image_size / (image_size + crop_pad) ) 为边长，从原图中 center_crop 出一个正方形
    def __init__(self, image_size=224, crop_pad=32):
        self.image_size = image_size
        self.crop_pad = crop_pad
        super(PaddedCenterCropTransformer, self).__init__()

    def run_transform(self, image):
        orig_height, orig_width, _ = image.shape
        padded_center_crop_size = int((float(self.image_size) /
                                   (self.image_size + self.crop_pad)) * \
                                  np.minimum(orig_height, orig_width))
        offset_height = ((orig_height - padded_center_crop_size) + 1) // 2
        offset_width = ((orig_width - padded_center_crop_size) + 1) // 2
        image = image[offset_height:offset_height + padded_center_crop_size,
                      offset_width:offset_width + padded_center_crop_size, :]
        image = image.astype(np.float32)
        return image


class _ChannelSwapTransformer(Transformer):
    def __init__(self, order, channel_index=0):
        self.order = order
        self.channel_index = channel_index
        super(_ChannelSwapTransformer, self).__init__()

    def run_transform(self, image):
        assert self.channel_index < len(image.shape), \
            "channel index is larger than image.dims"
        assert image.shape[self.channel_index] == len(self.order), \
            "the length of swap order != the number of channel:{}!={}" \
            .format(len(self.order), image.shape[self.channel_index])
        if self.channel_index == 0:
            image = image[self.order, :, :]
        elif self.channel_index == 1:
            image = image[:, self.order, :]
        elif self.channel_index == 2:
            image = image[:, :, self.order]
        else:
            raise ValueError(
                f"channel index: {self.channel_index} error in _ChannelSwapTransformer"
            )
        return image


class BGR2RGBTransformer(Transformer):
    def __init__(self, data_format="CHW"):
        if data_format == "CHW":
            self.transformer = _ChannelSwapTransformer((2, 1, 0))
        elif data_format == "HWC":
            self.transformer = _ChannelSwapTransformer((2, 1, 0), 2)
        else:
            raise ValueError(
                f"unsupported data_format: '{data_format}' in BGR2RGBTransformer"
            )

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class RGB2BGRTransformer(Transformer):
    def __init__(self, data_format="CHW"):
        if data_format == "CHW":
            self.transformer = _ChannelSwapTransformer((2, 1, 0))
        elif data_format == "HWC":
            self.transformer = _ChannelSwapTransformer((2, 1, 0), 2)
        else:
            raise ValueError(
                f"unsupported data_format: '{data_format}' in RGB2BGRTransformer"
            )

    def run_transform(self, data):
        return self.transformer.run_transform(data)


def rgb2bt601_full_range(r, g, b, single_channel=False):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    if not single_channel:
        u = -0.169 * r - 0.331 * g + 0.5 * b + 128
        v = 0.5 * r - 0.419 * g - 0.081 * b + 128
        return y, u, v
    else:
        return y


def rgb2bt601_video_range(r, g, b, single_channel=False):
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16
    if not single_channel:
        u = -0.148 * r - 0.291 * g + 0.439 * b + 128
        v = 0.439 * r - 0.368 * g - 0.071 * b + 128
        return y, u, v
    else:
        return y


class _ColorConvertTransformer(Transformer):
    def __init__(self, source_type, target_type, data_format="CHW"):
        # get source format and range
        source_format_range = source_type.split('_')
        source_format = source_format_range[0]
        source_range = source_format_range[1] \
            if len(source_format_range) == 2 else '255'
        target_type = target_type.upper()
        source_format = source_format.upper()
        data_format = data_format.upper()
        # get target format and range
        if target_type in ['YUV_BT601_VIDEO_RANGE', 'YUV_BT601_FULL_RANGE']:
            target_format = target_type
            target_range = '128'
        else:
            target_format_range = target_type.split('_')
            target_format = target_format_range[0]
            target_range = target_format_range[1] \
                if len(target_format_range) == 2 else '255'

        if source_format == target_format:
            self.transform_func = lambda img: img
        else:
            # split source input to r, g, b
            if source_format == 'RGB' and data_format == 'HWC':
                split_func = lambda img: (img[:, :, 0], img[:, :, 1], img[:, :,
                                                                          2])
            elif source_format == 'RGB' and data_format == 'CHW':
                split_func = lambda img: (img[0, :, :], img[1, :, :], img[
                    2, :, :])
            elif source_format == 'BGR' and data_format == 'HWC':
                split_func = lambda img: (img[:, :, 2], img[:, :, 1], img[:, :,
                                                                          0])
            elif source_format == 'BGR' and data_format == 'CHW':
                split_func = lambda img: (img[2, :, :], img[1, :, :], img[
                    0, :, :])
            else:
                ValueError(
                    "Unknown color convert source_format:{} or data_format{}, please check yaml"
                    .format(source_format, data_format))
            # convert r, g, b to yuv or gray
            if target_format == 'RGB':
                convert_func = lambda img: img
            elif target_format == 'BGR':
                convert_func = lambda img: (img[2], img[1], img[0])
            elif target_format == 'YUV444' or \
                    target_format == 'YUV_BT601_FULL_RANGE':
                convert_func = lambda img: rgb2bt601_full_range(*img)
            elif target_format == 'YUV_BT601_VIDEO_RANGE':
                convert_func = lambda img: rgb2bt601_video_range(*img)
            elif target_format == 'GRAY':
                convert_func = lambda img: rgb2bt601_full_range(
                    *img, single_channel=True)
            else:
                ValueError(
                    "Unknown color convert target_format:{}, please check yaml"
                    .format(target_format))
            # fuse convert result(b, g, r or y, u, v) to target output
            if data_format == 'HWC':
                if target_format == 'GRAY':
                    fuse_func = lambda img: img[:, :, np.newaxis]
                else:
                    fuse_func = lambda img: np.array(img).transpose((1, 2, 0))
            elif data_format == 'CHW':
                if target_format == 'GRAY':
                    fuse_func = lambda img: img[np.newaxis, :, :]
                else:
                    fuse_func = lambda img: np.array(img)
            self.transform_func = lambda img: fuse_func(
                convert_func(split_func(img)))
        # all the color convert operated on data range in [0, 255]
        self.source_offset = 128. if source_range == "128" else 0.
        self.target_offset = -128. if target_range == "128" else 0.
        super(_ColorConvertTransformer, self).__init__()

    def run_transform(self, image):
        image += self.source_offset
        converted_image = self.transform_func(image)
        converted_image += self.target_offset
        image = converted_image.astype(np.float32)
        return image


class RGB2GRAYTransformer(Transformer):
    def __init__(self, data_format):
        self.transformer = _ColorConvertTransformer('RGB', 'GRAY', data_format)

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class BGR2GRAYTransformer(Transformer):
    def __init__(self, data_format):
        self.transformer = _ColorConvertTransformer('BGR', 'GRAY', data_format)

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class RGB2GRAY_128Transformer(Transformer):
    def __init__(self, data_format):
        self.transformer = _ColorConvertTransformer('RGB', 'GRAY_128',
                                                    data_format)

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class RGB2YUV444Transformer(Transformer):
    def __init__(self, data_format):
        self.transformer = _ColorConvertTransformer('RGB', 'YUV444',
                                                    data_format)

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class BGR2YUV444Transformer(Transformer):
    def __init__(self, data_format):
        self.transformer = _ColorConvertTransformer('BGR', 'YUV444',
                                                    data_format)

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class BGR2YUV444_128Transformer(Transformer):
    def __init__(self, data_format):
        self.transformer = _ColorConvertTransformer('BGR', 'YUV444_128',
                                                    data_format)

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class RGB2YUV444_128Transformer(Transformer):
    def __init__(self, data_format):
        self.transformer = _ColorConvertTransformer('RGB', 'YUV444_128',
                                                    data_format)

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class BGR2YUVBT601VIDEOTransformer(Transformer):
    def __init__(self, data_format):
        self.transformer = _ColorConvertTransformer('BGR',
                                                    'YUV_BT601_Video_Range',
                                                    data_format=data_format)

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class RGB2YUVBT601VIDEOTransformer(Transformer):
    def __init__(self, data_format):
        self.transformer = _ColorConvertTransformer('RGB',
                                                    'YUV_BT601_Video_Range',
                                                    data_format=data_format)

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class YUVTransformer(Transformer):
    def __init__(self, color_sequence):
        self.color_sequence = color_sequence
        super(YUVTransformer, self).__init__()

    def _py_func(self, image, rgb_data=True):
        import math
        assert isinstance(image, np.ndarray), "Input must be numpy.ndarray"
        assert image.shape[2] == 3, "Input must be RGB or BGR."
        assert (np.max(image) < 256
                and np.min(image) >= 0), "Input must be between 0 and 255, \
            otherwise np.uint8 may cause unexpected problems."

        image = image.astype(np.uint8)
        if rgb_data:
            image = image[:, :, ::-1]
        img_h, img_w = image.shape[:2]
        uv_start_idx = img_h * img_w
        v_size = int(img_h * img_w / 4)
        # bgr -> yuv420sp
        img_yuv420sp = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
        img_yuv420sp = img_yuv420sp.flatten()
        # yuv420sp -> yuv444
        img_y = img_yuv420sp[:uv_start_idx].reshape((img_h, img_w, 1))
        uv_end_idx = uv_start_idx + v_size
        img_u = img_yuv420sp[uv_start_idx:uv_end_idx]
        img_u = img_u.reshape(int(math.ceil(img_h / 2.0)),
                              int(math.ceil(img_w / 2.0)), 1)
        img_u = np.repeat(img_u, 2, axis=0)
        img_u = np.repeat(img_u, 2, axis=1)
        v_start_idx = uv_start_idx + v_size
        v_end_idx = uv_start_idx + 2 * v_size
        img_v = img_yuv420sp[v_start_idx:v_end_idx]
        img_v = img_v.reshape(int(math.ceil(img_h / 2.0)),
                              int(math.ceil(img_w / 2.0)), 1)
        img_v = np.repeat(img_v, 2, axis=0)
        img_v = np.repeat(img_v, 2, axis=1)
        img_yuv444 = np.concatenate((img_y, img_u, img_v), axis=2)
        # uint8 --> float32
        img_yuv444 = img_yuv444.astype(np.float32)
        return img_yuv444

    def run_transform(self, data):
        data = self._py_func(data, self.color_sequence == "RGB")
        return data


class ReduceChannelTransformer(Transformer):
    def __init__(self, data_format="CHW"):
        self.data_format = data_format
        super(ReduceChannelTransformer, self).__init__()

    def run_transform(self, image):
        if self.data_format == "CHW":
            image = image[:1, ...]
        elif self.data_format == "HWC":
            image = image[..., :1]
        else:
            raise ValueError(
                "ReduceChannelTransformer only support CHW or HWC data format")
        return image


class BGR2NV12Transformer(Transformer):
    @staticmethod
    def mergeUV(u, v):
        if u.shape == v.shape:
            uv = np.zeros(shape=(u.shape[0], u.shape[1] * 2))
            for i in range(0, u.shape[0]):
                for j in range(0, u.shape[1]):
                    uv[i, 2 * j] = u[i, j]
                    uv[i, 2 * j + 1] = v[i, j]
            return uv
        else:
            raise ValueError("size of Channel U is different with Channel V")

    def __init__(self, data_format="CHW", cvt_mode='rgb_calc'):
        self.cvt_mode = cvt_mode
        self.data_format = data_format

    def rgb2nv12_calc(self, image):
        if image.ndim == 3:
            b = image[:, :, 0]
            g = image[:, :, 1]
            r = image[:, :, 2]
            y = (0.299 * r + 0.587 * g + 0.114 * b)
            u = (-0.169 * r - 0.331 * g + 0.5 * b + 128)[::2, ::2]
            v = (0.5 * r - 0.419 * g - 0.081 * b + 128)[::2, ::2]
            uv = self.mergeUV(u, v)
            yuv = np.vstack((y, uv))
            return yuv.astype(np.uint8)
        else:
            raise ValueError("image is not BGR format")

    def rgb2nv12_opencv(self, image):
        if image.ndim == 3:
            image = image.astype(np.uint8)
            height, width = image.shape[0], image.shape[1]
            yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape(
                (height * width * 3 // 2, ))
            y = yuv420p[:height * width]
            uv_planar = yuv420p[height * width:].reshape(
                (2, height * width // 4))
            uv_packed = uv_planar.transpose((1, 0)).reshape(
                (height * width // 2, ))
            nv12 = np.zeros_like(yuv420p)
            nv12[:height * width] = y
            nv12[height * width:] = uv_packed
            return nv12
        else:
            raise ValueError("image is not BGR format")

    def run_transform(self, image):
        if self.data_format == "CHW":
            image = np.transpose(image, (1, 2, 0))

        image_shape = image.shape[:-1]
        if image_shape[0] * image_shape[1] % 2 != 0:
            raise ValueError(
                f"Invalid odd shape: {image_shape[0]} x {image_shape[1]}, expect even number for height and width"
            )

        if self.cvt_mode == 'opencv':
            image = self.rgb2nv12_opencv(image)
        else:
            image = self.rgb2nv12_calc(image)
        return image


class RGB2NV12Transformer(Transformer):
    @staticmethod
    def mergeUV(u, v):
        if u.shape == v.shape:
            uv = np.zeros(shape=(u.shape[0], u.shape[1] * 2))
            for i in range(0, u.shape[0]):
                for j in range(0, u.shape[1]):
                    uv[i, 2 * j] = u[i, j]
                    uv[i, 2 * j + 1] = v[i, j]
            return uv
        else:
            raise ValueError("size of Channel U is different with Channel V")

    def __init__(self, data_format="CHW", cvt_mode='rgb_calc'):
        self.cvt_mode = cvt_mode
        self.data_format = data_format

    def rgb2nv12_calc(self, image):
        if image.ndim == 3:
            r = image[:, :, 0]
            g = image[:, :, 1]
            b = image[:, :, 2]
            y = (0.299 * r + 0.587 * g + 0.114 * b)
            u = (-0.169 * r - 0.331 * g + 0.5 * b + 128)[::2, ::2]
            v = (0.5 * r - 0.419 * g - 0.081 * b + 128)[::2, ::2]
            uv = self.mergeUV(u, v)
            yuv = np.vstack((y, uv))
            return yuv.astype(np.uint8)
        else:
            raise ValueError("image is not BGR format")

    def rgb2nv12_opencv(self, image):
        if image.ndim == 3:
            image = image.astype(np.uint8)
            height, width = image.shape[0], image.shape[1]
            yuv420p = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape(
                (height * width * 3 // 2, ))
            y = yuv420p[:height * width]
            uv_planar = yuv420p[height * width:].reshape(
                (2, height * width // 4))
            uv_packed = uv_planar.transpose((1, 0)).reshape(
                (height * width // 2, ))
            nv12 = np.zeros_like(yuv420p)
            nv12[:height * width] = y
            nv12[height * width:] = uv_packed
            return nv12
        else:
            raise ValueError("image is not BGR format")

    def run_transform(self, image):
        if self.data_format == "CHW":
            image = np.transpose(image, (1, 2, 0))

        image_shape = image.shape[:-1]
        if image_shape[0] * image_shape[1] % 2 != 0:
            raise ValueError(
                f"Invalid odd shape: {image_shape[0]} x {image_shape[1]}, expect even number for height and width"
            )

        if self.cvt_mode == 'opencv':
            image = self.rgb2nv12_opencv(image)
        else:
            image = self.rgb2nv12_calc(image)
        return image


class NV12ToYUV444Transformer(Transformer):
    def __init__(self, target_size, yuv444_output_layout="HWC"):
        super(NV12ToYUV444Transformer, self).__init__()
        self.height = target_size[0]
        self.width = target_size[1]
        self.yuv444_output_layout = yuv444_output_layout

    def run_transform(self, data):
        nv12_data = data.flatten()
        yuv444 = np.empty([self.height, self.width, 3], dtype=np.uint8)
        yuv444[:, :, 0] = nv12_data[:self.width * self.height].reshape(
            self.height, self.width)
        u = nv12_data[self.width * self.height::2].reshape(
            self.height // 2, self.width // 2)
        yuv444[:, :, 1] = Image.fromarray(u).resize((self.width, self.height),
                                                    resample=0)
        v = nv12_data[self.width * self.height + 1::2].reshape(
            self.height // 2, self.width // 2)
        yuv444[:, :, 2] = Image.fromarray(v).resize((self.width, self.height),
                                                    resample=0)
        data = yuv444.astype(np.uint8)
        if self.yuv444_output_layout == "CHW":
            data = np.transpose(data, (2, 0, 1))
        return data


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


class WarpAffineTransformer(Transformer):
    def __init__(self, input_shape, scale):
        self.input_height = input_shape[0]
        self.input_width = input_shape[1]
        self.scale = scale
        super(WarpAffineTransformer, self).__init__()

    def run_transform(self, data):
        origin_shape = data.shape[0:2]
        height, width = origin_shape
        new_height = int(height * self.scale)
        new_width = int(width * self.scale)
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
        trans_input = get_affine_transform(
            c, s, 0, [self.input_width, self.input_height])
        inp_image = cv2.warpAffine(data,
                                   trans_input,
                                   (self.input_width, self.input_height),
                                   flags=cv2.INTER_LINEAR)
        data = inp_image.astype(np.float32)

        return data


class F32ToS8Transformer(Transformer):
    def __init__(self):
        super(F32ToS8Transformer, self).__init__()

    def run_transform(self, data):
        data = data.astype(np.int8)
        return data


class F32ToU8Transformer(Transformer):
    def __init__(self):
        super(F32ToU8Transformer, self).__init__()

    def run_transform(self, data):
        data = data.astype(np.uint8)
        return data
