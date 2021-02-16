#!/bin/sh
python run.py test --dataset_test /media/Alpha/datasets/VoxCeleb2/Preprocessed/test/ \
                    --csv_test ./dataset/dataset_test.csv \
                    --device cuda \
                    --config ./configs/config.yaml
