#!/bin/sh
python run.py train --dataset_train /media/Alpha/datasets/VoxCeleb2/Videos/dev/ \
                    --dataset_test /media/Alpha/datasets/VoxCeleb2/Videos/test/ \
                    --csv_train ./dataset/dataset_dev.csv \
                    --csv_test ./dataset/dataset_test.csv \
                    --device cuda
