#!/bin/sh
python run.py test --dataset_test /home/kaan/datasets/VoxCeleb2/CropIdsPreprocessed/test/ \
                    --csv_test ./dataset/dataset_crop_test.csv \
                    --model /home/kaan/facial-reenactment/checkpoints/Generator_t20210323_0632_e005_i00109110.pth \
                    --config ./configs/config_creation.yaml
