#!/bin/sh
python run_creation.py infer \
                    --source "/media/Beta/facial-reenactment/datasets/VoxCeleb2/CropPreprocessed/test/id00017/5MkXgwdrmJw/1.png" \
                    --target "/media/Beta/facial-reenactment/datasets/VoxCeleb2/Videos/test/id01567/1Lx_ZqrK1bM/00001.mp4" \
                    --model "/media/Beta/facial-reenactment/models/creation/exp7/Generator_t20210311_2251_e029_i00037500.pth" \
                    --config ./configs/config_creation_model7.yaml
