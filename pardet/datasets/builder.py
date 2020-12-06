from torch.utils.data import DataLoader

from pardet.utils import Registry, build

DATASETS = Registry('DATASETS')
PIPELINES = Registry('PIPELINES')


def build_dataset(cfg):
    """Build detector."""
    return build(cfg, DATASETS)


def build_pipeline(cfg):
    """Build detector."""
    return build(cfg, PIPELINES)


def build_dataloader(dataset, batch_size, num_workers, shuffle=True):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )
    return data_loader
