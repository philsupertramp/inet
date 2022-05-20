#!/usr/bin/env python
import argparse
import datetime
import json
import os
import random as rng
from shutil import copyfile
from statistics import mean, median
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytz

from scripts.constants import CLASS_MAP
# Type definitions
from src.data.load_dataset import (LabeledFileDictType,
                                   load_labels_from_bbox_file)
from src.models.data_structures import BoundingBox


def load_labels_from_bbox_files(bbox_files: List[str], in_directory: str) -> LabeledFileDictType:
    labeled_files = dict()
    for elem in bbox_files:
        labeled_files.update(**load_labels_from_bbox_file(elem, in_directory))
    return labeled_files


def create_directory_structure(directory: str, class_names: Set[str]) -> Tuple[str, str, str]:
    sub_dirs = (
        os.path.join(directory, 'test'),
        os.path.join(directory, 'train'),
        os.path.join(directory, 'validation')
    )
    directories = [
        os.path.join(sub_dir, n)
        for n in class_names
        for sub_dir in sub_dirs
    ]

    for n in directories:
        os.makedirs(n, exist_ok=True)

    return sub_dirs


def split_labeled_files(files: LabeledFileDictType,
                        test_share: float, validation_share: float
                        ) -> Tuple[LabeledFileDictType, LabeledFileDictType, LabeledFileDictType]:

    filenames = np.array(list(files.keys()))
    num_files = len(filenames)
    file_indices = np.arange(0, num_files, dtype=int)
    np.random.shuffle(file_indices)

    # calculate exact numbers of samples for each data set
    number_test_elements = int(np.ceil(len(file_indices) * test_share))
    number_train_samples = len(file_indices) - number_test_elements
    number_validation_elements = int(np.ceil(number_train_samples * validation_share))

    # test/training split
    test_filenames = filenames[file_indices[:number_test_elements]].copy()
    training_indices = file_indices[number_test_elements:]

    # del file_array

    # validation/train split
    validation_filenames = filenames[training_indices[:number_validation_elements]].copy()
    train_filenames = filenames[training_indices[number_validation_elements:]].copy()

    return (
        {k: files[k] for k in test_filenames},
        {k: files[k] for k in train_filenames},
        {k: files[k] for k in validation_filenames}
    )


def spread_files(files: LabeledFileDictType, target_directory: str, label_names: List[str]) -> LabeledFileDictType:
    directories = [os.path.join(target_directory, n) for n in label_names]
    new_files = dict()
    for filepath, data in files.items():
        filename = filepath.split('/')[-1]
        target = directories[CLASS_MAP.get(data['label'])]
        path = os.path.join(target, filename)
        copyfile(filepath, path)
        new_files[filename] = data

    return new_files


def create_config_file(test_set, train_set, val_set, label_names, output_file):
    file_content = {
        'labels': list(label_names),
        'train': train_set,
        'validation': val_set,
        'test': test_set,
        'created_at': datetime.datetime.now(pytz.timezone('Europe/Berlin')).strftime('%m/%d/%Y %H:%M:%S')
    }
    with open(output_file, 'w') as f:
        json.dump(file_content, f, indent=2)


def create_dataset_structure(labeled_files: LabeledFileDictType, label_names, dir_name, test_split, val_split):
    """Plan:
    1. create structure
    2. split files into test, train, val
    3. move files
    """
    test_dir, train_dir, val_dir = create_directory_structure(dir_name, label_names)
    test_set, train_set, val_set = split_labeled_files(labeled_files, test_split, val_split)

    test_set = spread_files(test_set, test_dir, label_names)
    train_set = spread_files(train_set, train_dir, label_names)
    val_set = spread_files(val_set, val_dir, label_names)

    create_config_file(test_set, train_set, val_set, label_names, os.path.join(dir_name, 'dataset-structure.json'))


def get_genera_file_stats_for_directory(directory: str, labels) -> Dict[str, int]:
    files_per_genera = dict.fromkeys(labels, 0)

    for genera in files_per_genera.keys():
        files = os.listdir(os.path.join(directory, genera))
        files_per_genera[genera] = len(list(filter(lambda x: '.jpg' in x, files)))

    return files_per_genera


def test_output_directory(directory: str, labels) -> Tuple[Dict, Dict, Dict]:
    test_dir = os.path.join(directory, 'test')
    train_dir = os.path.join(directory, 'train')
    val_dir = os.path.join(directory, 'validation')

    def plot_stats(name, stats):
        plt.figure()
        plt.title(f'Distribution of samples per Order in the "{name}" set', fontsize=15)
        plt.bar(stats.keys(), stats.values())
        filename = os.path.join(directory, f'{name}-distribution')
        plt.savefig(f'{filename}.eps', bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{filename}.png', bbox_inches='tight', pad_inches=0)

    test_stats = get_genera_file_stats_for_directory(test_dir, labels)
    train_stats = get_genera_file_stats_for_directory(train_dir, labels)
    val_stats = get_genera_file_stats_for_directory(val_dir, labels)
    total_stats = {label: sum([test_stats[label], train_stats[label], val_stats[label]]) for label in labels}

    fig, ax = plt.subplots()

    sample_count = sum(total_stats.values())
    train_share = sum(train_stats.values()) / sample_count
    validation_share = sum(val_stats.values()) / sample_count
    test_share = sum(test_stats.values()) / sample_count

    ax.bar(labels, train_stats.values(), label=f'train set ({train_share*100:.0f}%)')
    ax.bar(labels, val_stats.values(), label=f'validation set ({validation_share*100:.0f}%)')
    ax.bar(labels, test_stats.values(), label=f'test set ({test_share*100:.0f}%)')
    fig.legend(
        loc='center left',  # Position of legend,
        borderaxespad=0.1,  # Small spacing around legend box
        bbox_to_anchor=(0.95, 0.5, 0, 0)
    )
    plt.title('Distributions for samples per order', fontsize=25)
    plt.savefig(os.path.join(directory, 'stacked-chart.eps'), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(directory, 'stacked-chart.png'), bbox_inches='tight', pad_inches=0)


    plot_stats('test', test_stats)
    plot_stats('training', train_stats)
    plot_stats('validation', val_stats)
    plot_stats('total data', total_stats)

    return test_stats, train_stats, val_stats


def bounding_box_stats(files):
    bbs = [BoundingBox(**elem['bbs']) for elem in files.values()]
    widths = [float(i.w) for i in bbs]
    heights = [float(i.h) for i in bbs]

    return {
        'max': {
            'width': max(widths),
            'height': max(heights),
        },
        'min': {
            'width': min(widths),
            'height': min(heights),
        },
        'avg': {
            'width': mean(widths),
            'height': mean(heights),
        },
        'median': {
            'width': median(widths),
            'height': median(heights),
        },
    }


def generate_statistics(files: Dict, target_directory: str, labels):
    bb_stats = bounding_box_stats(files)
    test, train, val = test_output_directory(target_directory, labels)

    print('Bounding Box statistics:')
    print(bb_stats)

    print('Test dataset stats:')
    print(test)

    print('Train dataset stats:')
    print(train)

    print('Validation dataset stats:')
    print(val)


if __name__ == '__main__':
    # Initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('bbox_label_files', nargs='+', help='name of bbox label file(s) to load labels from')
    parser.add_argument('-input_directory', help='name of the directory to load labeled images from')
    parser.add_argument('-output_directory', help='name of the directory to save labeled images to')
    parser.add_argument('-output_file', help='name of the output file')
    parser.add_argument('-val', '--validation-split', dest='val_split', default=0,
                        type=float, help='Share of samples for the validation set; in decimal format in range [0, 1]')
    parser.add_argument('-test', '--test-split', dest='test_split', default=0,
                        type=float, help='Share of samples for the test set; in decimal format in range [0, 1]')
    parser.add_argument('--reuse_directory', action='store_true', help='use existing files from this directory')
    parser.add_argument('-r', dest='reuse_directory', action='store_true', help='use existing files from this directory')
    parser.add_argument('--seed', default=42, type=int, help='Seed to use for rng')
    args = parser.parse_args()

    assert 0.0 <= args.test_split <= 1.0, f'Test share {args.test_split} not in range [0, 1]'
    assert 0.0 <= args.val_split <= 1.0, f'Validation share {args.val_split} not in range [0, 1]'

    rng.seed(args.seed)

    labeled_files = load_labels_from_bbox_files(args.bbox_label_files, args.input_directory)

    fn = args.output_file or './data.npy'

    labels = [genera.split('>')[0] for genera in CLASS_MAP.keys()]
    if args.output_directory:
        create_dataset_structure(labeled_files, labels, args.output_directory, args.test_split, args.val_split)

    generate_statistics(labeled_files, args.output_directory, labels)
