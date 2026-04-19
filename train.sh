EXPID=$(date +"%Y%m%d_%H%M%S")

python train.py \
  --config 'configs/train.yaml' \
  --output_dir './results' \
  --launcher none \
  --device cuda \
  --log_num "${EXPID}"
