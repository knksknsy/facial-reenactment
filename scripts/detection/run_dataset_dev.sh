#!/bin/sh
python run_detection.py dataset \
                        --source /home/kaan/datasets/FaceForensics/Videos/dev \
                        --output /home/kaan/datasets/FaceForensics/Preprocessed/dev \
                        --csv ./csv/faceforensics_dev.csv \
                        --num_videos 0 \
                        --device cuda \
                        --config ./configs/config_detection.yaml
