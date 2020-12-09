import os
import pickle
import numpy as np
from PIL import Image
from easydict import EasyDict

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

        self._img_idx = [ii for ii in range(len(self.image_names))]  # [:50 * 64]

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
        return len(self._img_idx)

    def evaluate(self, results, logger, threshold=0.5, metrics=('ma', 'acc', 'prec', 'rec', 'f1')):
        """Evaluation in PAR protocol."""
        probs, gt_labels = results["probs"], results["gt_labels"]
        pred_labels = np.array(probs) > threshold
        gt_labels = np.array(gt_labels)

        eps = 1e-20
        result = EasyDict()

        # label metrics
        gt_pos = np.sum((gt_labels == 1), axis=0).astype(float)  # TP + FN
        gt_neg = np.sum((gt_labels == 0), axis=0).astype(float)  # TN + FP
        true_pos = np.sum((gt_labels == 1) * (pred_labels == 1), axis=0).astype(float)  # TP
        true_neg = np.sum((gt_labels == 0) * (pred_labels == 0), axis=0).astype(float)  # TN
        false_pos = np.sum(((gt_labels == 0) * (pred_labels == 1)), axis=0).astype(float)  # FP
        false_neg = np.sum(((gt_labels == 1) * (pred_labels == 0)), axis=0).astype(float)  # FN
        label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
        label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
        label_ma = (label_pos_recall + label_neg_recall) / 2  # mean accuracy

        result.label_pos_recall = label_pos_recall
        result.label_neg_recall = label_neg_recall
        result.label_prec = true_pos / (true_pos + false_pos + eps)
        result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
        result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
                result.label_prec + result.label_pos_recall + eps)

        result.label_ma = label_ma
        result.ma = np.mean(label_ma)

        # instance metrics
        gt_pos = np.sum((gt_labels == 1), axis=1).astype(float)
        true_pos = np.sum((pred_labels == 1), axis=1).astype(float)
        intersect_pos = np.sum((gt_labels == 1) * (pred_labels == 1), axis=1).astype(float)  # true positive
        union_pos = np.sum(((gt_labels == 1) + (pred_labels == 1)), axis=1).astype(float)  # IOU

        instance_acc = intersect_pos / (union_pos + eps)
        instance_prec = intersect_pos / (true_pos + eps)
        instance_recall = intersect_pos / (gt_pos + eps)
        instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

        instance_acc = np.mean(instance_acc)
        instance_prec = np.mean(instance_prec)
        instance_recall = np.mean(instance_recall)
        instance_f1 = np.mean(instance_f1)

        result.acc = instance_acc
        result.prec = instance_prec
        result.rec = instance_recall
        result.f1 = instance_f1
        result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

        eval_res = dict()
        for metric in metrics:
            eval_res[metric] = result[metric]

        return eval_res
