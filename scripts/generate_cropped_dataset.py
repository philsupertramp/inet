#!/usr/bin/env python3
"""
Script to create a cropped version of the dataset generated using `process_files.py`.
Maintains pixels

Example:
    Input: 200px x 300px
    Bounding Box: 10px x 20px

    Output: 10px x 20px


For usage run `./generate_cropped_dataset.py -h`
"""
import argparse
import json
import os
from datetime import datetime

import pytz
from PIL import Image

from scripts.constants import CLASS_MAP
from scripts.process_files import create_directory_structure
from inet.models.data_structures import BoundingBox

if __name__ == '__main__':
    ## Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', help='path root to extract images from nested directory "data"')
    parser.add_argument('-m', '--multi-threading', dest='multi_threading', action='store_true', default=False, help='Use multiple threads file moving')
    args = parser.parse_args()

    ## input directory name
    input_dir = os.path.join(args.root_dir, 'data')
    ## output directory name
    output_dir = os.path.join(args.root_dir, 'cropped-data')
    ## data set configuration file
    dataset_file = os.path.join(input_dir, 'dataset-structure.json')

    with open(dataset_file) as fp:
        dataset = json.load(fp)

    labels = {genera.split('>')[0] for genera in CLASS_MAP.keys()}
    create_directory_structure(output_dir, labels)
    new_data = {
        'train': {},
        'test': {},
        'validation': {},
        'labels': list(labels),
        'created_at': datetime.now(pytz.timezone('Europe/Berlin')).strftime('%m/%d/%Y %H:%M:%S')
    }
    for set_name in ['train', 'test', 'validation']:
        for filename, data in dataset.get(set_name, {}).items():
            new_filename = os.path.join(input_dir, set_name, filename.split('_')[1], filename)
            img = Image.open(new_filename)
            bb = BoundingBox(**data['bbs'])
            new_filename = new_filename.replace(input_dir, output_dir).split('/')
            new_filename[-1] = f'cropped-{new_filename[-1]}'
            new_filename = '/'.join(new_filename)
            img.crop(
                (
                    int(bb.x_min * img.width/100.),
                    int(bb.y_min * img.height/100.),
                    int(bb.x_max * img.width/100.),
                    int(bb.y_max * img.height/100.),
                )
            ).save(new_filename)
            new_data[set_name][new_filename] = {'label': data['label']}

    with open(os.path.join(output_dir, 'dataset-structure.json'), 'w') as fp:
        json.dump(new_data, fp)
