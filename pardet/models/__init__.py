# Copyright (c) 2020, All rights reserved.
# Author: wduo, wduo@163.com
# pardet-pytorch for Pedestrian Attribute Recognition.
from .backbones import ResNet50
from .par_detectors import StrongBaseline
from .losses import CEL_Sigmoid
from .builder import build_parnet, build_loss

__all__ = ["build_parnet", "build_loss"]
