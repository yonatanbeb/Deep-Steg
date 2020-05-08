# Deep-Steg


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install modules in Project.

```bash
pip install -r requirements.txt
```

## Usage

To train Deep, Auto Encoder based, Steganography models, type:

```bash
. ./train.sh
```

If Steganography models already exist, see evaluation by typing:

```bash
. ./evaluate.sh
```

To view datasets:

```bash
# for original fashion_mnist dataset
python view.py 

# for encoded and auto encoded datasets
python view.py [-v | --version] {encoded / auto_encoded} [-z | -o]
```