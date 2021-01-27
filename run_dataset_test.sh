#!/bin/sh
python run.py dataset --source /media/Alpha/datasets/VoxCeleb2/Videos/test/ \
                      --output /media/Alpha/datasets/VoxCeleb2/Preprocessed/test/ \
                      --csv ./dataset/dataset_test.csv \
                      --size 0 \
                      --device cuda
