#!/bin/sh
python run.py dataset creation \
                      --source /media/Alpha/datasets/VoxCeleb2/Videos/dev/ \
                      --output /home/kaan/datasets/VoxCeleb2/CropPreprocessed/dev/ \
                      --csv ./dataset/dataset_crop_dev.csv \
                      --num_videos 0 \
                      --device cuda \
                      --config ./configs/config_creation.yaml
