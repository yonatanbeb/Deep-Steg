#!/bin/bash

cd models
# run scripts that evaluates the AutoEncoder models and Classifier
python model_evaluation.py
cd ../datasets
# run script that displays an example of the steganography methods
python steg_example.py
cd ..