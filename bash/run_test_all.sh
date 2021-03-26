#!/bin/sh
python run.py test --dataset_test /home/kaan/datasets/VoxCeleb2/CropIdsPreprocessed/test/ \
                    --csv_test ./dataset/dataset_crop_test.csv \
                    --device cuda \
                    --num_workers 8 \
                    --config ./configs/config_creation.yaml
