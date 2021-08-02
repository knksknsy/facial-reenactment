#!/bin/sh
python run_detection.py infer \
                    --source ./inputs/a.png \
                    --model ./checkpoints/model.pth \
                    --device cuda \
                    --config ./configs/config_detection.yaml
