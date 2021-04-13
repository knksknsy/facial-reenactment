#!/bin/sh
python run_detection.py dataset \
                        --source /home/kaan/datasets/FaceForensics/Videos/test \
                        --output /home/kaan/datasets/FaceForensics/Preprocessed/test \
                        --csv ./csv/faceforensics_test.csv \
                        --num_videos 0 \
                        --device cuda \
                        --config ./configs/config_detection.yaml
