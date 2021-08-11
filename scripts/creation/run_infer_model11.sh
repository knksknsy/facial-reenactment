#!/bin/sh
python run_creation.py infer \
                    --source "/Volumes/1\ TB\ Data/facial-reenactment/datasets/VoxCeleb2/CropPreprocessed/test/id00017/5MkXgwdrmJw/1.png" \
                    --target "/Volumes/1\ TB\ Data/facial-reenactment/datasets/VoxCeleb2/Videos/test/id01567/1Lx_ZqrK1bM/00001.mp4" \
                    --model "/Volumes/1\ TB\ Data/facial-reenactment/models/creation/exp11/Generator_t20210401_1236_e029_i00150000.pth" \
                    --config ./configs/config_creation_model11.yaml
