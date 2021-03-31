#!/bin/sh
python run.py infer creation \
                    --source /home/kaan/datasets/VoxCeleb2/CropIdsPreprocessed/test/id00017/5MkXgwdrmJw/1.png \
                    --target /home/kaan/datasets/VoxCeleb2/CropIdsPreprocessed/test/id01567/1Lx_ZqrK1bM/1.png \
                    --model /media/Beta/facial-reenactment/experiments/c_128/5_final_its_1250_pg_70_reflect_plateau/checkpoints/Generator_t20210330_1211_e029_i00037500.pth \
                    --config ./configs/config_creation.yaml
