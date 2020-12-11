# Copyright (c) 2020, All rights reserved.
# Author: wduo, wduo@163.com
# pardet-pytorch for Pedestrian Attribute Recognition.
import os
import random
import pickle
import numpy as np
from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

group_order = [7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 9, 10, 11, 12, 1, 2, 3, 0, 4, 5, 6]


def save_split(split_name, dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, split_name + '.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


def generate_split(split_name, attr_names, image_names, labels, reorder, weights):
    dataset = EasyDict()
    dataset.split_name = split_name
    dataset.image_names = image_names
    dataset.labels = labels
    dataset.weights = weights
    dataset.attr_names = attr_names
    if reorder:
        dataset.labels = dataset.labels[:, np.array(group_order)]
        dataset.attr_names = [dataset.attr_names[i] for i in group_order]

    return dataset


def generate_data_splits(save_dir, reorder):
    """
    create a dataset description file, which consists of images, labels etc.
    """
    # raw PA100k_data
    pa100k_data = loadmat(os.path.join(save_dir, 'annotation.mat'))

    train_image_names = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_names = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_names = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    trainval_image_names = train_image_names + val_image_names

    train_labels = np.concatenate([[pa100k_data['train_label'][i]] for i in range(80000)], axis=0)
    val_labels = np.concatenate([[pa100k_data['val_label'][i]] for i in range(10000)], axis=0)
    test_labels = np.concatenate([[pa100k_data['test_label'][i]] for i in range(10000)], axis=0)
    trainval_labels = np.concatenate((train_labels, val_labels), axis=0)

    attr_names = [pa100k_data['attributes'][i][0][0] for i in range(26)]

    # idx
    train_idx = np.arange(0, 80000)  # np.array(range(80000))
    val_idx = np.arange(0, 10000)  # np.array(range(80000, 90000))
    test_idx = np.arange(0, 10000)  # np.array(range(90000, 100000))
    trainval_idx = np.arange(0, 90000)

    # weights
    train_weights = np.mean(train_labels[train_idx], axis=0).astype(np.float32)
    val_weights = np.mean(val_labels[val_idx], axis=0).astype(np.float32)
    test_weights = np.mean(test_labels[test_idx], axis=0).astype(np.float32)
    trainval_weights = np.mean(trainval_labels[trainval_idx], axis=0).astype(np.float32)

    # generate splits
    train_split_name = 'PA100k_train_split'
    train_dataset = generate_split(train_split_name, attr_names,
                                   train_image_names, train_labels, reorder, train_weights)

    val_split_name = 'PA100k_val_split'
    val_dataset = generate_split(val_split_name, attr_names,
                                 val_image_names, val_labels, reorder, val_weights)

    test_split_name = 'PA100k_test_split'
    test_dataset = generate_split(test_split_name, attr_names,
                                  test_image_names, test_labels, reorder, test_weights)

    trainval_split_name = 'PA100k_trainval_split'
    trainval_dataset = generate_split(trainval_split_name, attr_names,
                                      trainval_image_names, trainval_labels, reorder, trainval_weights)

    # save
    save_split(train_split_name, train_dataset, save_dir)
    save_split(val_split_name, val_dataset, save_dir)
    save_split(test_split_name, test_dataset, save_dir)
    save_split(trainval_split_name, trainval_dataset, save_dir)


if __name__ == "__main__":
    save_dir = '/pardet/data/PA100K/annotation/'
    reoder = True
    generate_data_splits(save_dir, reorder=True)
