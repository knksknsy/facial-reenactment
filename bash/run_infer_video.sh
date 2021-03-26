#!/bin/sh
python run.py infer --source ./inputs/a.png \
                    --target ./inputs/b.mp4 \
                    --model ./checkpoints/Generator.pth \
                    --device cuda \
                    --config ./configs/config_creation.yaml
