#!/bin/sh
set -xe
if [ ! -f train.py ]; then
    echo "Please make sure you run this from STT's top level directory."
    exit 1
fi;

checkpoint_dir="$HOME/.local/share/stt/antonija"

# Force only one visible device because we have a single-sample dataset
# and when trying to run on multiple devices (like GPUs), this will break
export CUDA_VISIBLE_DEVICES=0

python -m coqui_stt_training.train \
  --alphabet_config_path "data/antonija/cro_alphabet.txt" \
  --show_progressbar false \
  --train_files data/antonija/antonija.csv \
  --test_files data/antonija/antonija.csv \
  --train_batch_size 1 \
  --test_batch_size 1 \
  --n_hidden 100 \
  --epochs 200 \
  --checkpoint_dir "$checkpoint_dir" \
  "$@"
