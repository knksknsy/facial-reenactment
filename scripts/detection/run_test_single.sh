#!/bin/sh
python run_creation.py test \
                    --dataset_test /home/kaan/datasets/FaceForensics/Preprocessed/test/ \
                    --csv_test ./csv/faceforensics_test.csv \
                    --model /media/Alpha/experiments/detection/final_11b_experiment_e21_34_lr_1e-6/checkpoints/SiameseResNet_t20210420_1700_e034_i00019890.pth \
                    --config /media/Alpha/experiments/detection/final_11b_experiment_e21_34_lr_1e-6/config_detection.yaml
