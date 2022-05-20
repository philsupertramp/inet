#!/usr/bin/env python
"""
Use this script to copy the labeled images from the mounted volume
into the dedicated structure.
"""

import os

import numpy as np
from PIL import Image


def extract_file_name(elem):
    fn = elem['file_upload']
    return fn[fn.find('-')+1:]


def get_label(fn):
    prefix = int(fn)
    if prefix in [i for i in range(184, 418)]:
        return 0
    elif prefix in [i for i in range(738, 771)]:
        return 1
    elif prefix in [i for i in range(848, 2277)]:
        return 2
    elif prefix in [i for i in range(2304, 2594)]:
        return 3
    elif prefix in [i for i in range(507, 670)]:
        return 4


if __name__ == '__main__':
    import argparse

    # Initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('input_directory', help='name of directory to load images from')
    args = parser.parse_args()

    content = None
    file_names = list(filter(lambda x: '.jpg' in x, list(os.listdir(args.input_directory))))

    X = []
    y = []
    print(f'Will process {len(file_names)} files for directory {args.input_directory}.')
    for fn in file_names:
        X.append(np.array(Image.open(os.path.join(args.input_directory, fn)).resize((224, 224))))
        y.append(get_label(fn.split('_')[0]))

    with open('data-prep-full.npy', 'wb') as f:
        np.save(f, X)
        np.save(f, y)
