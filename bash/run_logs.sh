#!/bin/sh
python run.py logs --logs_dir "/media/Beta/facial-reenactment/experiments/gs_64/" \
                    --plots ./configs/plots.json \
                    --config ./configs/config.yaml \
                    --overwrite_logs
