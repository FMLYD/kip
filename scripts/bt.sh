export CUDA_VISIBLE_DEVICES=3

python  benchmark_kpi.py \
  --model KPI \
  --dataset_name  blood_transfusion  \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.1\
  --stop 10 \
  --k 30\
  --metric nan_manhattan \
  --weights uniform

  python  benchmark_kpi.py \
  --model KPI \
  --dataset_name  blood_transfusion  \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.2\
  --stop 10 \
  --sigma "0.1,0.3,0.5,0.7,1,3,5" \
  --k 60\
  --metric nan_manhattan \
  --weights uniform

  python  benchmark_kpi.py \
  --model KPI \
  --dataset_name  blood_transfusion  \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.3\
  --stop 10 \
  --k 70\
  --metric nan_manhattan \
  --weights uniform

  python  benchmark_kpi.py \
  --model KPI \
  --dataset_name  blood_transfusion  \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.4\
  --stop 10 \
  --k 50\
  --metric nan_manhattan \
  --weights uniform
