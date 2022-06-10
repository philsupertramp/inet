#!/usr/bin/env python
"""
Use this script to copy the labeled images from the mounted volume
into the dedicated structure.
"""

import json
import os
from shutil import copy
from typing import Dict

import numpy as np

from scripts.helpers import ProgressBar, ThreadWithReturnValue


def extract_file_name(elem: Dict) -> str:
    """
    Extract file name from element dictionary
    :param elem: element dictionary
    :return: file name
    """
    if 'file_upload' in elem:
        fn = elem['file_upload']
    else:
        fn = elem['data']['image']
    return fn.split('/')[-1]


def get_directory_from_prefix(in_dir, prefix):
    """
    search function for directory with prefix `prefix` within `in_dir`
    :param in_dir: parent directory
    :param prefix: search term to look for
    :return: path to target directory
    """
    directory = os.listdir(in_dir)
    directory = [i for i in filter(lambda x: prefix in x, directory)][0]
    return os.path.join(in_dir, directory)


def move_file(elem, input_dir, target_dir):
    """
    moves files from one directory into another
    :param elem: files to move
    :param input_dir: source to move from
    :param target_dir: target to move files to
    :return: list of files in target directory
    """
    fn = extract_file_name(elem)
    target_file = os.path.join(target_dir, fn)
    copy(
        os.path.join(get_directory_from_prefix(input_dir, fn.split('_')[0]), fn.split('_')[-1]),
        target_file
    )
    return target_file


def process_in_multi_threads(file_content, input_dir, target_dir) -> None:
    """
    Moves files `file_content` from `input_dir` to `target_dir` using multiple threads
    :param file_content: list of file names
    :param input_dir: source directory
    :param target_dir: target directory
    :return:
    """
    # increase this number to increase the number of used threads
    max_running_threads = 10

    list_len = len(file_content)
    number_iterations = int(np.ceil(list_len / max_running_threads))+1
    element_index = 0
    pb = ProgressBar(number_iterations)
    generated_file_names = []
    for i in range(number_iterations):
        threads = []
        for thread_id in range(max_running_threads):
            if element_index >= list_len:
                break

            threads.append(
                ThreadWithReturnValue(
                    target=move_file,
                    args=(file_content[element_index], input_dir, target_dir)
                )
            )
            element_index += 1

        for thread in threads:
            thread.start()

        for thread in threads:
            generated_file_names.append(thread.join())
        pb.step(i)

    assert element_index == len(set(generated_file_names)), 'Inconsistency in copying files.'
    pb.done()


def process_in_single_thread(file_content, input_dir, target_dir) -> None:
    """
    Copies files `file_content` from `input_dir` to `target_dir` using a single thread.

    :param file_content: files to move
    :param input_dir: source directory
    :param target_dir: target directory
    :return:
    """
    list_len = len(file_content)
    pb = ProgressBar(list_len)
    generated_file_names = []
    element_index = 0
    for element_index, elem in enumerate(file_content):
        generated_file_names.append(move_file(elem, input_dir, target_dir))
        pb.step(element_index)

    assert element_index == len(generated_file_names), 'Inconsistency in copying files.'
    pb.done()


if __name__ == '__main__':
    import argparse

    # Initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='+', help='name of label file to load images from')
    parser.add_argument('input_dir', help='path to load images from')
    parser.add_argument('target_dir', help='path to save images to')
    parser.add_argument('-m', '--multi-threading', dest='multi_threading', action='store_true', default=False, help='Use multiple threads file moving')
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    filename = args.filename[0]
    with open(filename) as json_file:
        content = json.load(json_file)

    if args.multi_threading:
        process_in_multi_threads(content, args.input_dir, args.target_dir)
    else:
        process_in_single_thread(content, args.input_dir, args.target_dir)
