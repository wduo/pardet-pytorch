# Copyright (c) 2020, All rights reserved.
# Author: wduo, wduo@163.com
# pardet-pytorch for Pedestrian Attribute Recognition.
from pardet.utils import get_root_logger
from pardet.datasets import build_dataset, build_dataloader
from pardet.runner import build_optimizer, Runner, EvalHook


def train_detector(model,
                   dataset,
                   cfg,
                   evaluate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.batchsize,
            cfg.data.workers,
            cfg.data.shuffle) for ds in dataset
    ]

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    runner.timestamp = timestamp

    # register hooks
    optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # register eval hooks
    if evaluate:
        test_dataset = build_dataset(cfg.data.test)
        test_dataloader = build_dataloader(dataset=test_dataset, batch_size=1,
                                           num_workers=cfg.data.workers, shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        runner.register_hook(EvalHook(test_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
