#!/bin/sh
python run.py test --dataset_test /media/Alpha/datasets/VoxCeleb2/Videos/test/ \
                    --csv_test ./dataset/dataset_test.csv \
                    --model ./checkpoints/Generator.pth \
                    --device cuda \
                    --config ./configs/config.yaml
