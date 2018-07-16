#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

import chainer
from chainer.training import extensions
import chainercv

#here = osp.dirname(osp.abspath(__file__))

def get_data():
    from dataset import "HOGE"

    dataset_train = Hogedataset(split='train')
    iter_train = chainer.iterators.SerialIterator(
        dataset_train, batch_size=1)

    dataset_valid_raw = Hogedataset(split='val', return_image=True)
    iter_valid_raw = cahiner.iterators.SerialIterator(
        dataset_valid_raw, batch_size=1, repeat=False, shuffle=False)

    dataset_valid = HogeDataset(split='val')
    iter_valid = chainer.iterators.SerialIterator(
        dataset_valid, batch_size=1, repeat=False, shuffle=False)

    return iter_train, iter_valuid, iter_valid_raw

def get trainer(optimizer, iter_train, iter_valid, iter_valid_raw,
                args):
    model = optimizer.target

    updater = chainer.training.StandardUpdater(
        iter_train, optimizer, device=args.gpu)

    trainer = chainer.training.Trainer(
        updater, (args.max_iteration, 'iteration'), out=args.out)

    #trainer.extend

    #trainer.extend(extensions.ProgressReport(args.__dict__))

    # trainer.extend(extensions.LogReport(
    #     trigger=(args.interval_print, 'iteration')))
    # trainer.extend(extensions.PrintReport(
    #     ['epoch', 'iteration', 'elapsed_time',
    #      'main/loss', 'validation/main/miou']))




def main()
