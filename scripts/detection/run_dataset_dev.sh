#!/bin/sh
python run_detection.py dataset \
                        --source /home/kaan/datasets/FaceForensics/Videos/dev \
                        --output /home/kaan/datasets/FaceForensics/PreprocessedExt/dev \
                        --csv ./csv/faceforensics_ext_dev.csv \
                        --num_videos 0 \
                        --max_frames 40 \
                        --device cuda \
                        --config ./configs/config_detection.yaml
