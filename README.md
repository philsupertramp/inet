# thesisphilipp

## Thesis: Machine Learning Methods for Localization and Classification of Insects in Images


## Description
This project contains all content and things around the thesis to obtain the degree Bachelor of Science.

For now the repository contains the `./docs` directory for the theoretical part of the thesis, a demo webapp and several scripts used during the preparation of the first dataset as well
as the first design of model architectures.
The directory `src/` contains different model architectures `src/models/` that are all designed to solve the supervised learning
tasks of classification and bounding box regression, either individually or combined.

## Visuals

### Data Augmentation
![Augmentation examples](./notebooks/visualizations/classification-test.png)

## Installation

### Prerequesites:
- An account to access the VPN, as well as an account to access the NAS is required
- `openfortivpn`
- set environment variables according to [`mount_volumes.sh`](mount_volumes.sh)
- `python >= 3.8, virtualenv`

## Usage
### Datasets
#### iNaturalist
**use available `*.npy` files:**
All available `*.npy` files follow the same structure
```text
X
labels
bounding-boxes
```
to extract the data use
```python
import numpy as np

with open('./data.npy', 'rb') as f:
    X = np.load(f, allow_pickle=True)
    labels = np.load(f, allow_pickle=True)
    bbs = np.load(f, allow_pickle=True)
```
#### recreate a used training set:
```shell
$ sudo ./scripts/mount_volumes.sh on
$ python -m scripts.reuse_labels bounding-boxes-2022-02-12-14-33.json mnt/KInsektDaten/data/iNat/train_Insecta/ data/iNat/storage
```
#### generate a training set:

1. Set environment variables `USERNAME` (username on NAS), `PASSWORD` (password for NAS), `VPN_USER` and `VPN_PASSWD` accordingly
2. Run:
```shell
$ sudo ./mount_volumes.sh on
```
3. To generate a dataset from the source `mnt/KInsektDaten/data/iNat/train_Insecta/`:
```shell
$ python -m scripts.preselect_files --seed 42 -g 20 -s 25 -rng -l ../mnt/KInsektDaten/data/iNat/train_Insecta/ ../data/iNat/
```
for more options see `-h`.

4. Upload the files within the (default) target directory `./data/iNat/storage` into ["Label-Studio"](https://labelstudio-kinsekt.app.datexis.com) and generate bounding boxes.

optionally Launch [LabelStudio]()
```shell
$ docker run -it -p 8080:8080 -v $PWD/data/iNat:/label-studio/data -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data heartexlabs/label-studio:latest
```
5. Create labels for image files
6. Export the labels from LStudio
7. Generate file structure for train, test and validation sets by running
```shell
$ python -m scripts.process_files -input_directory data/iNat/storage -output_directory data/iNat/data -test 0.1 -val 0.2 bounding-boxes-2022-02-12-14-33.json
```
8. Generate cropped dataset for classification task
```shell
$ python -m scripts.generate_cropped_dataset data/iNat/
```

## Features

### Tex docs
To build the `./doc` directory see [`docs/tex/README.md`](docs/tex/README.md).

### Webapp
To run the webapp see [`src/app/README.md`](src/webapp/README.md)


# TODO

## Support
In case you need help setting up the project or run into issues please create a ticket within the repositories issue tracker

## Authors and acknowledgment

## License

## Project status
The project is currently Work in Progress (WIP).
