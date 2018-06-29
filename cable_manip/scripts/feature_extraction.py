#!/usr/bin/env python

import argparse
import os
import os.path as osp

import chainer
import numpy as np
import skimage.io

import vgg16


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
    img = skimage.io.imread('image.png')
    #input = input[np.newaxis, :, :, :]

    model = vgg16.VGG16()
    #forward
    with chainer.no_backprop_mode():
        input_data = chainer.Variable(img)
        model(input_data)

if __name__ == '__main__':
    extraction_tmp()
