#!/bin/sh
python run_creation.py dataset \
                      --source /media/Alpha/datasets/VoxCeleb2/Videos/test/ \
                      --output /home/kaan/datasets/VoxCeleb2/CropIdsPreprocessed/test/ \
                      --csv ./csv/voxceleb_crop_test.csv \
                      --num_videos 0 \
                      --num_frames 2 \
                      --device cuda \
                      --config ./configs/config_creation.yaml
