#!/usr/bin/env python
"""
Data augmentation using https://augmentor.readthedocs.io/en/master
"""
import argparse
import os
from shutil import rmtree

from Augmentor import Pipeline

from scripts.helpers import move_files

## parent directory of current directory
root_directory = os.path.join(os.path.dirname(__file__), '..')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_directory', help='name of input directory to use input files from')
    parser.add_argument('-seed', default=42, help='Seed to use')
    parser.add_argument('-n', '--number_samples', type=int, default=5_000, help='Number of total samples to produce')
    parser.add_argument(
        '-c', '--cleanup', action='store_true',
        help='Cleanup the directory afterwards. This will move the augmented data into the provided `input_directory`'
    )

    args = parser.parse_args()

    seed = args.seed
    number_samples = args.number_samples
    input_directory = args.input_directory
    output_directory = os.path.join(input_directory, 'output')
    cleanup = args.cleanup

    p = Pipeline(input_directory, output_directory=output_directory)
    p.rotate90(probability=0.5)
    p.rotate270(probability=0.5)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.skew(probability=0.5)
    p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
    p.random_brightness(0.1, 0.8, 1.5)
    p.set_seed(seed)

    p.sample(number_samples)

    if cleanup:
        move_files([os.path.join(output_directory, f) for f in os.listdir(output_directory)], input_directory)
        rmtree(output_directory)
