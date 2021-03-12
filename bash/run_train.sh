#!/bin/sh
CONFIG="./configs/config.yaml"

CHECKPOINT_DIR="$(cat $CONFIG | grep checkpoint_dir)"
CHECKPOINT_DIR="${CHECKPOINT_DIR#*": "}" # Remove yaml property; Keep value

DO=1
while [ $DO -eq 1 ]
    do
        # Start training
        python run.py train --config $CONFIG
        # Get stdout and stderr
        RESULT=$?

        # stderr
        # Catch exception: CUDA out of memory
        if [ $RESULT -ne 0 ]; then
            # Get latest checkpoint
            LATEST_CHECKPOINT="$(ls -t $CHECKPOINT_DIR | head -n 1)"
            LATEST_CHECKPOINT="${LATEST_CHECKPOINT#"Discriminator_"}"
            
            # Add LATEST_CHECKPOINT to config.yaml
            sed -i "s/continue_id.*/continue_id: $LATEST_CHECKPOINT/" $CONFIG

            echo "Restart training with latest checkpoint_id: $LATEST_CHECKPOINT"
        
        # stdout
        # Exit while loop if training finished without exceptions
        else
            DO=0
            echo "Training finished"
        fi
    done
