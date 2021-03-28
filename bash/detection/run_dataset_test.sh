#!/bin/sh
python run.py dataset detection \
                        --source /home/kaan/datasets/FaceForensics/Videos/test \
                        --output /home/kaan/datasets/FaceForensics/Preprocessed/test \
                        --csv ./dataset/faceforensics_test.csv \
                        --num_videos 0 \
                        --device cuda \
                        --config ./configs/config_detection.yaml
