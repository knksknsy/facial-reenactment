#!/bin/sh
python run.py dataset detection \
                        --source /home/kaan/datasets/FaceForensics/Videos/dev \
                        --output /home/kaan/datasets/FaceForensics/Preprocessed/dev \
                        --csv ./dataset/faceforensics_dev.csv \
                        --num_videos 0 \
                        --device cuda \
                        --config ./configs/config_detection.yaml
