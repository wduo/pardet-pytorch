# import os
# print(os.path.realpath(__file__))

from torch import nn

from pardet.utils import Registry, build_from_cfg

BOCKBONES = Registry('BACKBONES')
CLASSIFIERS = Registry('CLASSIFIERS')
PARNETS = Registry('PARNETS')


def build_bockbone(cfg):
    """Build detector."""
    return build(cfg, BOCKBONES)


def build_classifier(cfg):
    """Build detector."""
    return build(cfg, CLASSIFIERS)


def build_parnet(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    return build(cfg, PARNETS)


def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)
