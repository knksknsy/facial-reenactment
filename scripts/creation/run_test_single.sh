#!/bin/sh
python run_creation.py test \
                    --dataset_test /home/kaan/datasets/VoxCeleb2/CropIdsPreprocessed/test/ \
                    --model /media/Beta/facial-reenactment/experiments/creation/c_128/7_final_its_5000_pg_70_reflect_plateau/checkpoints/Generator_t20210401_1236_e029_i00150000.pth \
                    --num_workers_test 1 \
                    --conv_blocks_d 4 \
                    --image_size 128 \
                    --channels 3 \
                    --shuffle_test \
                    --batch_size_test 8 \
                    --log_freq_test 1 \
                    --config ./configs/config_creation.yaml
