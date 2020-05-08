#!/bin/bash

# run scripts that train AutoEncoder models and Classifier
python auto_encode_dataset.py
cd datasets
# run script to update encoded and autoencoded datasets
python dataset_generator.py
cd ..