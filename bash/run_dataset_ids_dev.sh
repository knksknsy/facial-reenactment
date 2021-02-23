#!/bin/sh
python run.py dataset --source /media/Alpha/datasets/VoxCeleb2/Videos/dev/ \
                      --output /media/Alpha/datasets/VoxCeleb2/IdsPreprocessed/dev/ \
                      --csv ./dataset/dataset_ids_dev.csv \
                      --num_videos 0 \
                      --device cuda \
                      --config ./configs/config.yaml \
                      --vox_ids     id02149 id08892 id08339 id04305 id05345 id07658 id08623 id02152 id02191 id01883 \
                                    id03037 id08290 id07151 id06838 id01081 id06848 id02081 id07306 id08804 id07335 \
                                    id04459 id04941 id05568 id05262 id00597 id03931 id06432 id04524 id09070 id06143 \
                                    id08259 id00991 id08454 id09231 id03540 id09029 id07520 id03904 id08747 id02037 \
                                    id03609 id04737 id00519 id07550 id01685 id08649 id02038 id06497 id03262 id06753 \
                                    id02475 id04245 id03003 id02724 id04690 id00515 id01316 id04838 id03455 id01578 \
                                    id00735 id05966 id06399 id02455 id03345 id06061 id05372 id06328 id05308 id00763 \
                                    id04675 id07991 id05581 id04049 id03822 id07255 id08549 id00104 id05314 id01317 \
                                    id04545 id05772 id05710 id08801 id01870 id08481 id00570 id05441 id03100 id07685 \
                                    id00892 id04027 id04132 id08917 id00328 id06377 id07560 id01455 id00359 id01187
