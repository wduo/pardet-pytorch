import os
import pickle
import numpy as np
from PIL import Image

import torch.utils.data as data

from .pipelines import Compose
from .builder import DATASETS


@DATASETS.register_module()
class PA100K(data.Dataset):
    def __init__(self, ann_file, img_prefix, pipeline=None, target_transform=None):
        dataset_info = pickle.load(open(ann_file, 'rb+'))
        self.description = dataset_info.split_name
        self.image_names = dataset_info.image_names
        self.labels = dataset_info.labels
        self.weights = dataset_info.weights
        self.attr_names = dataset_info.attr_names

        self.img_prefix = img_prefix
        self.pipeline = Compose(pipeline)
        self.target_transform = target_transform

    def __getitem__(self, idx):
        img_name, label = self.image_names[idx], self.labels[idx]
        img_path = os.path.join(self.img_prefix, img_name)
        img = Image.open(img_path)

        if self.pipeline is not None:
            img = self.pipeline(img)
        label = label.astype(np.float32)
        if self.target_transform is not None:
            label = self.pipeline(label)

        data_item = dict(img=img, gt_label=label, img_name=img_name)
        return data_item

    def __len__(self):
        return len(self.image_names)
