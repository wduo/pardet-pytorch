# import os
# print(os.path.realpath(__file__))

from pardet.utils import Registry, build

BOCKBONES = Registry('BACKBONES')
CLASSIFIERS = Registry('CLASSIFIERS')
PARNETS = Registry('PARNETS')
LOSSES = Registry('LOSSES')


def build_bockbone(cfg):
    """Build detector."""
    return build(cfg, BOCKBONES)


def build_classifier(cfg):
    """Build detector."""
    return build(cfg, CLASSIFIERS)


def build_parnet(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    return build(cfg, PARNETS)


def build_loss(cfg):
    """Build detector."""
    return build(cfg, LOSSES)
