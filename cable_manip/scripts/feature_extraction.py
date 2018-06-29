#!/usr/bin/env python

import argparse
import os
import os.path as osp

import chainer
import numpy as np
import skimage.io

import vgg16

import chainer.serializers

def extraction():
    #input
    for file in args.img_files:
        img = skimage.io.imread(file)
        input = input[np.newaxis, :, :, :]
        if args.gpu >= 0:
            input = chainer.cuda.to_gpu(input)

    #forward
    with chainer.no_backprop_mode():
        input = chainer.Variable(input)
        model(input)


def extraction_tmp():
    #input
    mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))

    img = skimage.io.imread('image.png')
    img = img.astype(np.float32)
    img -= mean_bgr
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :]

    model = vgg16.VGG16()
    #chainer.serializers.load_npz(vgg16.pretrained_model, model)

    #forward
    with chainer.no_backprop_mode():
        input_data = chainer.Variable(img)
        with chainer.using_config('train', False):
            feature = model(input_data)
    return feature


if __name__ == '__main__':
    extraction_tmp()
