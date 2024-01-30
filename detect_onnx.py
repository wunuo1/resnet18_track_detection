# Copyright (c) 2021 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import os
import click
import logging

from horizon_tc_ui import HB_ONNXRuntime
from horizon_tc_ui.utils.tool_utils import init_root_logger, on_exception_exit
from horizon_tc_ui.data.imagenet_val import imagenet_val
from PIL import Image
from PIL import ImageDraw

# from preprocess import infer_image_preprocess
# from postprocess import postprocess
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys
sys.path.append("../../../01_common/python/data/")
from transformer import *
from dataloader import *

class RGB2NV12Transform(object):
    def mergeUV(self, u, v):
        if u.shape == v.shape:
            uv = np.zeros(shape=(u.shape[0], u.shape[1] * 2))
            for i in range(0, u.shape[0]):
                for j in range(0, u.shape[1]):
                    uv[i, 2 * j] = u[i, j]
                    uv[i, 2 * j + 1] = v[i, j]
            return uv
        else:
            raise ValueError("size of Channel U is different with Channel V")
    def __call__(self, img):
        if img.mode == "RGB":
            img = np.array(img)
            r = img[:, :, 0]
            g = img[:, :, 1]
            b = img[:, :, 2]
            y = (0.299 * r + 0.587 * g + 0.114 * b)
            u = (-0.169 * r - 0.331 * g + 0.5 * b + 128)[::2, ::2]
            v = (0.5 * r - 0.419 * g - 0.081 * b + 128)[::2, ::2]
            uv = self.mergeUV(u, v)
            yuv = np.vstack((y, uv))
            return Image.fromarray(yuv.astype(np.uint8))
        else:
            raise ValueError("image is not RGB format")

class NV12ToYUV444Transformer(object):
    def __init__(self, target_size, yuv444_output_layout="HWC"):
        super(NV12ToYUV444Transformer, self).__init__()
        self.height = target_size[0]
        self.width = target_size[1]
        self.yuv444_output_layout = yuv444_output_layout

    def __call__(self, data):
        data = np.array(data)
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
        return Image.fromarray(data)

def inference(sess, image_name, input_layout, input_offset):
    if input_layout is None:
        logging.warning(f"input_layout not provided. Using {sess.layout[0]}")
        input_layout = sess.layout[0]

    # preprocess
    image = Image.open(image_name)
    imagedraw = ImageDraw.Draw(image)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        RGB2NV12Transform(),
        NV12ToYUV444Transformer((224,224),'HWC'),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[0.003921 , 0.003921 , 0.003921 ])  # 标准化
    ])
    transformed_image = image_transform(image)
    image_numpy = transformed_image.numpy()
    # image_numpy = image_numpy.transpose((2, 1, 0))
    
    image_numpy = np.expand_dims(image_numpy, axis=0)
    input_name = sess.input_names[0]
    output_name = sess.output_names
    output = sess.run(output_name, {input_name: image_numpy},
                      input_offset=input_offset)

    # postprocess
    prob = np.squeeze(output[0])
    x = int(112 * prob[0] + 112 * (640 / 224))
    y = int(112 - 112 * prob[1])
    print(prob)
    print(x)
    print(y)

    for i in range(x, x + 10):
        for j in range(y, y + 10):
            imagedraw.point((i, j), (255,0,0))
    image.save("./test.jpg")
    # idx = np.argsort(-prob)
    # top_five_label_probs = [(idx[i], prob[idx[i]], imagenet_val[idx[i]])
    #                         for i in range(5)]


@click.version_option(version="1.0.0")
@click.command()
@click.option('-m', '--model', type=str, help='Input onnx model(.onnx) file')
@click.option('-i', '--image', type=str, help='Input image file.')
@click.option('-y',
              '--input_layout',
              type=str,
              default="",
              help='Model input layout')
@click.option('-o',
              '--input_offset',
              type=str,
              default=128,
              help='input inference offset.')
@click.option('-c',
              '--color_sequence',
              type=str,
              default=None,
              help='Color sequence')
@on_exception_exit
def main(model, image, input_layout, input_offset, color_sequence):
    init_root_logger("inference",
                     console_level=logging.INFO,
                     file_level=logging.DEBUG)
    if color_sequence:
        logging.warning("option color_sequence is deprecated.")
    sess = HB_ONNXRuntime(model_file=model)
    sess.set_dim_param(0, 0, '?')
    inference(sess, image, input_layout, input_offset)


if __name__ == '__main__':
    main()
