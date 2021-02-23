#!/bin/sh
python run.py test --dataset_test /media/Alpha/datasets/VoxCeleb2/Preprocessed/test/ \
                    --csv_test ./dataset/dataset_test.csv \
                    --model /media/Alpha/facial-reenactment/experiment_3/3_1_epoch_0_29_lr_desc/checkpoints/Generator_t20210215_1711_e029_i00547830.pth \
                    --test_batch_size 1 \
                    --device cuda \
                    --config ./configs/config.yaml
