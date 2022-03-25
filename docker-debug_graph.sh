docker run \
  --rm -it --init \
  --gpus=all \
  --ipc=host \
  --network=host \
  --volume="$PWD/scripts/models:/app/scripts/models" \
  --volume="$PWD/scripts/logs:/app/scripts/logs" \
  babyai_kg \
  python train_rl.py --env BabyAI-GoTo-v0 --model debug_graph --procs 2 --val-episodes 2 --gpus 2 --sgr 0 --spr 0 --ws 3 --gpu_ids 0,1 --master_addr tulsi --log-interval 2 --save-interval 2
