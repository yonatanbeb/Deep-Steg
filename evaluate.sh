#!/bin/bash

cd models
python model_evaluation.py
cd ../datasets
python steg_example.py
cd ..