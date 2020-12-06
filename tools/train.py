# Copyright (c) 2020, All rights reserved.
# Author: wduo, wduo@163.com
# pardet-pytorch for Pedestrian Attribute Recognition.
import time
import argparse
import copy
import os.path as osp
import os
import sys

sys.path.insert(0, osp.join('..', os.getcwd()))

from pardet.models import build_parnet
from pardet.datasets import build_dataset
from pardet.utils import Config, DictAction
from pardet.utils import set_random_seed, mkdir_or_exist, get_root_logger
from pardet.apis import train_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--no-evaluate', action='store_true',
                        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int, help='number of gpus to use '
                                                     '(only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+', help='ids of gpus to use '
                                                                   '(only applicable to non-distributed training)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                             'in xxx=yyy format will be merged into config file.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

    # model
    # model = build_parnet(cfg, train_cfg=cfg["train_cfg"], test_cfg=cfg["test_cfg"])
    model = build_parnet(cfg.model).cuda()

    # dataset
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    train_detector(
        model,
        datasets,
        cfg,
        evaluate=(not args.no_evaluate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
