#!/bin/sh
python run.py dataset --source /media/Alpha/datasets/VoxCeleb2/Videos/dev/ \
                      --output /media/Alpha/datasets/VoxCeleb2/Preprocessed/dev/ \
                      --csv ./dataset/dataset_dev.csv \
                      --size 0 \
                      --device cuda
