import os
import pickle
import numpy as np
from PIL import Image

import torch.utils.data as data

from .pipelines import Compose
from .builder import DATASETS


@DATASETS.register_module()
class PA100K(data.Dataset):
    def __init__(self, split, ann_file, pipeline=None, target_transform=None):
        dataset_info = pickle.load(open(ann_file, 'rb+'))
        img_id = dataset_info.image_name
        attr_label = dataset_info.label
        assert split in dataset_info.partition.keys(), f'split {split} is not exist'

        self.dataset = 'PA100k'
        self.pipeline = Compose(pipeline)
        self.target_transform = target_transform
        self.root_path = dataset_info.root
        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)
        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]

    def __getitem__(self, index):
        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath)

        if self.transform is not None:
            img = self.pipeline(img)
        gt_label = gt_label.astype(np.float32)
        if self.target_transform is not None:
            gt_label = self.pipeline(gt_label)

        return img, gt_label, imgname

    def __len__(self):
        return len(self.img_id)
