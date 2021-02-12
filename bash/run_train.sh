#!/bin/sh
python run.py train --dataset_train /home/kaan/datasets/VoxCeleb2/Preprocessed/dev/ \
                    --dataset_test /home/kaan/datasets/VoxCeleb2/Preprocessed/test/ \
                    --csv_train ./dataset/dataset_dev.csv \
                    --csv_test ./dataset/dataset_test.csv \
                    --device cuda \
                    --config ./configs/config.yaml
