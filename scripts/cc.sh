export CUDA_VISIBLE_DEVICES=2

python  exper_standard_kip.py \
  --model KPI \
  --dataset_name concrete_compression \
  --outpath './results' \
  --lr 0.01\
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.1 \
  --stop 20 \
  --sigma "0.1,0.3,0.5,0.7,0.9,1,3,5,7"\
  --k 4\
  --metric nan_manhattan \
  --weights distance

  python  exper_standard_kip.py \
  --model KPI \
  --dataset_name concrete_compression \
  --outpath './results' \
  --lr 0.01\
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.2 \
  --stop 20 \
  --sigma "0.1,0.3,0.5,0.7,0.9,1,3,5,7" \
  --k 3\
  --metric nan_manhattan \
  --weights distance

  python  exper_standard_kip.py \
  --model KPI \
  --dataset_name concrete_compression \
  --outpath './results' \
  --lr 0.01\
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.3 \
  --stop 20 \
  --sigma "0.1,0.3,0.5,0.7,0.9,1,3,5,7" \
  --k 30\
  --metric nan_manhattan \
  --weights distance

  python  exper_standard_kip.py \
  --model KPI \
  --dataset_name concrete_compression \
  --outpath './results' \
  --lr 0.01\
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.4 \
  --stop 20 \
  --sigma "0.1,0.3,0.5,0.7,0.9,1,3,5,7"\
  --k 90\
  --metric nan_manhattan \
  --weights distance
  
