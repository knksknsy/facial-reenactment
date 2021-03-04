#!/bin/sh
python run.py train --dataset_train /home/kaan/datasets/VoxCeleb2/CropIdsPreprocessed/dev/ \
                    --dataset_test /home/kaan/datasets/VoxCeleb2/CropIdsPreprocessed/test/ \
                    --csv_train ./dataset/dataset_crop_ids_dev.csv \
                    --csv_test ./dataset/dataset_crop_test.csv \
                    --device cuda \
                    --config ./configs/config.yaml
