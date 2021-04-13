#!/bin/sh
python run_creation.py dataset \
                      --source /media/Alpha/datasets/VoxCeleb2/Videos/dev/ \
                      --output /home/kaan/datasets/VoxCeleb2/CropPreprocessed/dev/ \
                      --csv ./csv/voxceleb_crop_dev.csv \
                      --num_videos 0 \
                      --device cuda \
                      --config ./configs/config_creation.yaml
