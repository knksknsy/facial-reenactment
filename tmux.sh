#!/bin/bash
SESSION="$USER"
WINDOW="$SESSION:1"

tmux -2 new-session -s "$SESSION" -d 

tmux new-window -t "$WINDOW" -n 'Logs'
tmux split-window -t "$WINDOW" -h -p 20
tmux split-window -t "$WINDOW" -v -p 20 -d 'watch -n 1 sensors'
tmux select-window -t "$WINDOW"
tmux send-keys 'watch -n 1 nvidia-smi' C-m

tmux -2 attach-session -t "$SESSION"

