#!/bin/sh
python run_detection.py logs \
                    --logs_dir "/media/Alpha/experiments/detection/" \
                    --plots ./configs/plots_detection.json \
                    --config ./configs/config_detection.yaml
