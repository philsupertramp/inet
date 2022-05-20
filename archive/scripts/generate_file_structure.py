import os
from uuid import UUID

import numpy as np
from PIL import Image

from src.data.load_dataset import datasets_from_file

BASE_DIR = '../train-data'


def convert_datafile_to_file_structure(filename, label_names):
    train = datasets_from_file(filename, ('X', 'labels'), lambda x: x, test_share=0.0)
    directories = [os.path.join(BASE_DIR, n) for n in label_names]
    for n in directories:
        os.makedirs(n, exist_ok=True)

    for index, (x, y) in enumerate(train):
        for batch_index in range(len(x)):
            target = directories[np.argmax(y[batch_index])]
            uuid = UUID(int=index * len(x) + batch_index)
            path = os.path.join(target, f'{uuid}.jpg')
            Image.fromarray(np.array(x[batch_index])).save(path)


if __name__ == '__main__':
    labels = {'Lepidoptera', 'Coleoptera', 'Ordonata', 'Hymenoptera', 'Hemiptera'}
    convert_datafile_to_file_structure('../data-prep-full.npy', labels)
