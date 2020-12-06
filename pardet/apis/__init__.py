# Copyright (c) 2020, All rights reserved.
# Author: wduo, wduo@163.com
# pardet-pytorch for Pedestrian Attribute Recognition.
from .train import train_detector
from .test import single_gpu_test

__all__ = ['train_detector', 'single_gpu_test']
