#!/bin/bash

python auto_encode_dataset.py
cd datasets
python dataset_generator.py
python steg_example.py
