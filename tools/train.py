import os
import time
import argparse
import numpy as np
import os.path as osp
from PIL import Image

import torch

from pardet.models import build_parnet
from pardet.datasets import build_dataset
from pardet.utils import Config, DictAction
from pardet.utils import mkdir_or_exist, get_root_logger, collect_env

from pardet.apis import set_random_seed, train_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path',
                        default='../configs/strongbaseline.py')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
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
        cfg.work_dir = osp.join('../work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # create work_dir
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # init_dist(args.launcher, **cfg.dist_params)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # model
    # body = build_bodynet(cfg_body, train_cfg=cfg_body["train_cfg"], test_cfg=cfg_body["test_cfg"])
    model = build_parnet(cfg.model).cuda()

    # dataset
    datasets = [build_dataset(cfg.data.train)]

    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

    #
    img_path = '/opt/project/tests/data/0.2.jpg'
    im = Image.open(osp.abspath(img_path)).convert('RGB')
    epoch = 0
    max_epochs = 100
    while 1:
        # cuda0 = torch.device('cuda:0')
        x = np.random.randn(8, 3, 256, 192)
        x = torch.from_numpy(x)
        x = x.float().cuda()
        x_out = model(x)

        epoch += 1
        if epoch >= max_epochs:
            break

    pass


if __name__ == '__main__':
    main()
