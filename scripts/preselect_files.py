#!/usr/bin/env python
"""
Script to prepare iNaturalist image files based on
input parameters
"""
import argparse
import os
import random as rng
import shutil
from shutil import rmtree
from typing import Dict, List, Set, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.constants import (CLASS_MAP, CLASS_RANGES, COLEOPTERA_STR,
                               HEMIPTERA_STR, HYMENOPTERA_STR, LEPIDOPTERA_STR,
                               ORDONATA_STR)
from scripts.helpers import ProgressBar, ThreadWithReturnValue, decision

## Holds available ID ranges
USED_ID_RANGES = {}


def get_n_random_elements(container: List[Union[str, int]], num_elements: int) -> Set[Union[str, int]]:
    """
    Getter for `num_elements` random elements out of `container`
    :param container: the container to extract the elements from
    :param num_elements: number of random elements to extract
    :return: list of `num_elements` random elements out of `container`
    """
    output = set()
    while len(output) < num_elements:
        output |= {rng.choice(container)}
    return output


def init_random_dataset_ids(num_elements: int) -> List[str]:
    """

    :param num_elements:
    :return:
    """
    coleoptera_choices = get_n_random_elements(USED_ID_RANGES.get(COLEOPTERA_STR), num_elements)
    hymenoptera_choices = get_n_random_elements(USED_ID_RANGES.get(HYMENOPTERA_STR), num_elements)
    lepidoptera_choices = get_n_random_elements(USED_ID_RANGES.get(LEPIDOPTERA_STR), num_elements)
    ordonata_choices = get_n_random_elements(USED_ID_RANGES.get(ORDONATA_STR), num_elements)
    hemiptera_choices = get_n_random_elements(USED_ID_RANGES.get(HEMIPTERA_STR), num_elements)

    return [
        str(i).rjust(5, '0')
        for i in coleoptera_choices | hymenoptera_choices | lepidoptera_choices | ordonata_choices | hemiptera_choices
    ]


def calculate_stats(combined_list: List[Union[str, int]]) -> None:
    """

    :param combined_list:
    :return:
    """
    stats = dict.fromkeys(CLASS_MAP.keys(), 0)

    for elem in set(combined_list):
        x = int(elem)
        for key, value in USED_ID_RANGES.items():
            stats[key] += x in value

    print('Distribution of Species per Order:')
    print(stats)
    plt.title('Distribution of Species per Order in base set')
    plt.bar(stats.keys(), stats.values())
    plt.savefig('../base-set-distribution.eps', bbox_inches='tight', pad_inches=0)
    plt.savefig('../base-set-distribution.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def decide_image(img_path: str) -> bool:
    """

    :param img_path:
    :return:
    """
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    dec = decision('Do you want to use it?')
    plt.close()
    return dec


def get_directory_by_prefix(prefix: str, base_dir: str) -> str:
    """

    :param prefix:
    :param base_dir:
    :return:
    """
    files = list(filter(lambda x: prefix in x, os.listdir(base_dir)))
    if len(files) < 1:
        raise FileNotFoundError(f'Can\'t find {prefix}')
    return base_dir + files[0]


def process_directory(dir_prefix: str, base_dir: str, out_dir: str, requires_decisions: bool = False,
                      num_samples: int = 30) -> List[str]:
    """
    Copies `num_samples` files from input to target directory.
    Allows to require approval for each sample by adding `requires_decision=True`

    :param dir_prefix: prefix of source/input directory, e.g. 00420
    :param base_dir: root directory of source
    :param out_dir: target/output directory
    :param requires_decisions: if `True` asks for approval before copying each sample
    :param num_samples: required amount of samples
    :return: List of new file locations
    """
    directory = get_directory_by_prefix(dir_prefix, base_dir)
    files = os.listdir(directory)
    files = filter(lambda x: '.jpg' in x, files)
    files = np.array([f'{directory}/{f}' for f in files])

    assert len(files) >= num_samples, f'Cannot proceed, source directory "{directory}" does not contain enough samples.'

    np.random.shuffle(files)
    stored_files = []

    # !!! replace=False for unique elements!
    choices = np.random.choice(files, num_samples, replace=False)

    for index, choice in enumerate(choices):
        file_path = choice
        file_name = file_path.split('/')[-1]
        taxonomy = file_path.split('/')[-2].split('_')
        out_name = f'{taxonomy[0]}_{taxonomy[4]}_{taxonomy[5]}_{file_name}'

        if not requires_decisions or decide_image(file_path):
            output_path = f'{out_dir}/{out_name}'
            shutil.copy(file_path, output_path)
            stored_files.append(output_path)
    return stored_files


def create_local_copy(combined_list: List[Union[str, int]], base_dir: str, storage_dir: str, requires_decisions: bool, num_elements: int) -> List[str]:
    """
    Creates local copy of `num_elements` samples per record in `combined_list` from samples in `base_dir` into `storage_dir`.

    Uses multi-threading.

    :param combined_list: List of order directories
    :param base_dir: parent directory of input files
    :param storage_dir: storage directory to create copy in
    :param requires_decisions: if `True` asks for approval before adding a sample
    :param num_elements: required amount of samples per order
    :return: List of new file locations
    """
    print(f'\rStart processing file directories.')
    file_list = []
    list_len = len(combined_list)

    max_running_threads = 10
    number_iterations = int(np.ceil(list_len / max_running_threads))
    element_index = 0
    pb = ProgressBar(number_iterations)

    for i in range(number_iterations):
        threads = []
        for thread_id in range(max_running_threads):
            if element_index >= list_len:
                break

            threads.append(
                ThreadWithReturnValue(
                    target=process_directory,
                    args=(combined_list[element_index], base_dir),
                    kwargs=dict(
                        out_dir=storage_dir,
                        requires_decisions=requires_decisions,
                        num_samples=num_elements
                    )
                )
            )
            element_index += 1

        for thread in threads:
            thread.start()

        for thread in threads:
            file_list.extend(thread.join())
        pb.step(i)

    pb.done()
    print('Done processing.')
    return file_list


def process_directories(
        combined_list: List[Union[str, int]], base_dir: str, out_dir: str = '../data/iNat',
        requires_decisions: bool = False, num_elements: int = 30, move_files_only: bool = False) -> None:
    """
    Method to generate a local copy of the data set.

    :param combined_list: List of sample directories
    :param base_dir: root directory of source files
    :param out_dir: target directory
    :param requires_decisions: if `True` requires approval before adding a sample
    :param num_elements: required number of samples per order/species
    :param move_files_only: if `True` moves
    :return:
    """
    # clean target directory
    if not move_files_only:
        rmtree(out_dir, ignore_errors=True)

    storage_dir = os.path.join(out_dir, 'storage')
    directories = [
        storage_dir
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    create_local_copy(combined_list, base_dir, storage_dir, requires_decisions, num_elements)
    print(f'\rDone moving files.')


def test_dataset(combined_list: List[Union[str, int]], base_dir: str, expected_num_elements: int) -> None:
    """
    Verifies the dataset for consistency/integrity

    :param combined_list:
    :param base_dir:
    :param expected_num_elements:
    :return:
    """
    generated_files = sorted(os.listdir(os.path.join(base_dir, 'storage')))

    elements = {}
    print('Testing for integrity of the dataset...')

    for file in generated_files:
        f = file.split('_')
        order_id = f[1]
        if elements.get(order_id):
            elements[order_id].append(f[-1])
        else:
            elements[order_id] = [f[-1]]
    count = 0
    for key, values in elements.items():
        count += len(values)

    if count != expected_num_elements:
        print(f'{expected_num_elements - count} Missing')
    print(f'Total amount of genera: {len(set(combined_list))}')


def get_id_range_for_search_term(file_list: List[str], search_terms: List[str]) -> List[int]:
    """
    Filters given file name list by a list of search terms

    :param file_list: list of file names
    :param search_terms: list of search terms to search for
    :return: list of ids matching the search terms
    """
    return [int(i.split('_')[0]) for i in filter(lambda x: all([term in x for term in search_terms]), file_list)]


def get_id_ranges_from_input_directory(root_directory: str) -> Dict[str, List[int]]:
    """

    :param root_directory: root directory to get labels from
    :return: a dictionary containing species-ID ranges for all genera
    """
    files_in_dir = os.listdir(root_directory)
    return {
        COLEOPTERA_STR: get_id_range_for_search_term(files_in_dir, [COLEOPTERA_STR]),
        HEMIPTERA_STR: get_id_range_for_search_term(files_in_dir, HEMIPTERA_STR.split('>')),
        HYMENOPTERA_STR: get_id_range_for_search_term(files_in_dir, HYMENOPTERA_STR.split('>')),
        LEPIDOPTERA_STR: get_id_range_for_search_term(files_in_dir, [LEPIDOPTERA_STR]),
        ORDONATA_STR: get_id_range_for_search_term(files_in_dir, [ORDONATA_STR]),
    }


def scan_input_directory(root_directory: str) -> None:
    """
    Prints statistics from input directory
    :param root_directory:
    :return:
    """
    id_ranges = get_id_ranges_from_input_directory(root_directory)
    print('Species per Genera')
    print('=' * 20)
    print('\n'.join([f'{key}: {len(value)}' for key, value in id_ranges.items()]))
    print('=' * 20)
    number_files = dict.fromkeys(id_ranges.keys(), 0)
    print('\rScanning directories...', end='')
    for key, values in id_ranges.items():
        print(f'\rScanning directory {key}...', end='')
        for value in values:
            number_files[key] += len(os.listdir(get_directory_by_prefix(str(value).zfill(5), root_directory)))
        print(f'\rScanning directory {key} done!', end='')
    print('Files per Genera')
    print('\n'.join([f'{key}: {value}' for key, value in number_files.items()]))
    print('=' * 20)


if __name__ == '__main__':
    # Initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('input_directory', help='name of input directory to use input files from')
    parser.add_argument('output_directory', help='name of output directory to store files in')
    parser.add_argument('-r', dest='reuse_ids', action='store_true', help='reuse predefined id ranges')
    parser.add_argument('-rng', dest='rng_samples', action='store_true', help='Chose random samples')
    parser.add_argument('-g', '--genera', dest='num_elements', default=10, help='Number of samples per genera')
    parser.add_argument('-s', '--species', dest='num_samples', default=10, help='Number of samples per species')
    parser.add_argument('--seed', default=42, help='A seed to use for underlying random number generators')
    parser.add_argument('--scan-only', dest='scan', action='store_true',
                        help='Display input directory information then exit')
    parser.add_argument('--test-only', dest='test', action='store_true',
                        help='Test output directory then exit')
    parser.add_argument('--move-only', dest='move_only', action='store_true',
                        help='Only spread objects from nested "tmp" directory')
    args = parser.parse_args()

    number_elements = int(args.num_elements)
    number_samples = int(args.num_samples)
    root_dir = args.input_directory
    output_dir = args.output_directory
    reuse_ids = args.reuse_ids
    use_rng_samples = args.rng_samples
    seed = int(args.seed)
    scan = args.scan
    test = args.test
    move_only = args.move_only

    # set seeds
    rng.seed(seed)
    np.random.seed(seed)

    if scan:
        scan_input_directory(root_dir)
        exit(0)

    # use predefined id ranges, or scrape the root directory
    USED_ID_RANGES = CLASS_RANGES.copy()
    if not reuse_ids:
        USED_ID_RANGES = get_id_ranges_from_input_directory(root_dir)

    # extract random file directory names
    random_list = init_random_dataset_ids(number_elements)
    rng.shuffle(random_list)

    calculate_stats(random_list)

    # generate file structure
    process_directories(
        random_list,
        root_dir,
        out_dir=output_dir,
        requires_decisions=not use_rng_samples,
        num_elements=number_samples,
        move_files_only=move_only
    )

    # test the output
    ds_size = len(random_list) * number_samples
    test_dataset(random_list, output_dir, ds_size)
