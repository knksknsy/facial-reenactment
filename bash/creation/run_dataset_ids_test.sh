#!/bin/sh
python run.py dataset creation \
                      --source /media/Alpha/datasets/VoxCeleb2/Videos/test/ \
                      --output /home/kaan/datasets/VoxCeleb2/CropIdsPreprocessed/test/ \
                      --csv ./dataset/dataset_crop_test.csv \
                      --num_videos 0 \
                      --num_frames 2 \
                      --device cuda \
                      --config ./configs/config_creation.yaml
