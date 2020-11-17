from torch import nn
from torch.utils.data import DataLoader

from pardet.utils import Registry, build_from_cfg

DATASETS = Registry('DATASETS')
PIPELINES = Registry('PIPELINES')


def build_dataset(cfg):
    """Build detector."""
    return build(cfg, DATASETS)


def build_pipeline(cfg):
    """Build detector."""
    return build(cfg, PIPELINES)


def build_dataloader(train_set, batchsize, shuffle=True):
    data_loader = DataLoader(
        dataset=train_set,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
    )
    return data_loader


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
