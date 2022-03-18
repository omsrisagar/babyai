docker run \
  --rm -it --init \
  --gpus=all \
  --ipc=host \
  --network=host \
  --volume="$PWD/scripts/models:/app/scripts/models" \
  --volume="$PWD/scripts/logs:/app/scripts/logs" \
  babyai_kg \
  python train_rl.py --env BabyAI-GoTo-v0 --model goto_graph_ddp --pretrained-model goto_graph_ddp --procs 60 --val-episodes 60 --gpus 2 --sgr 0 --spr 0 --ws 3 --gpu_ids 0,1 --master_addr localhost
