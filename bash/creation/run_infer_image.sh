#!/bin/sh
python run.py infer creation \
                    --source ./inputs/a.png \
                    --target ./inputs/b.png \
                    --model ./checkpoints/Generator.pth \
                    --device cuda \
                    --config ./configs/config_creation.yaml
