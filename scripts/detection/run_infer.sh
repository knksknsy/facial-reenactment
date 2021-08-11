#!/bin/sh
python run_detection.py infer \
                    --source "/Volumes/1\ TB\ Data/facial-reenactment/datasets/FaceForensics/Videos/test/manipulated_sequences/NeuralTextures/c23/videos/132_007.mp4" \
                    --model "/Volumes/1\ TB\ Data/facial-reenactment/models/detection/exp6/SiameseResNet_t20210416_2034_e019_i00010710.pth" \
                    --config ./configs/config_detection_model6.yaml
