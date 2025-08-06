export CUDA_VISIBLE_DEVICES=2
python  benchmark_kpi.py \
  --model KPI \
  --dataset_name qsar_biodegradation \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 50 \
  --batch_size 128 \
  --p $p \
  --sigma "0.1,0.3,0.5,0.7,1,3,5" \
  --k 2\
  --metric nan_manhattan \
  --weights distance
  
  python  benchmark_kpi.py \
  --model KPI \
  --dataset_name qsar_biodegradation \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 50 \
  --batch_size 128 \
  --p $p \
  --sigma "0.1,0.3,0.5,0.7,1,3,5" 、
  --k 3\
  --metric nan_manhattan \
  --weights distance

  
  python  benchmark_kpi.py \
  --model KPI \
  --dataset_name qsar_biodegradation \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 50 \
  --batch_size 128 \
  --p $p \
  --sigma "0.1,0.3,0.5,0.7,1,3,5" \
  --k 3\
  --metric nan_manhattan \
  --weights distance
  
  python  benchmark_kpi.py \
  --model KPI \
  --dataset_name qsar_biodegradation \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 50 \
  --batch_size 128 \
  --p 5 \
  --sigma "0.1,0.3,0.5,0.7,1,3,5" \
  --k 5\
  --metric nan_manhattan \
  --weights distance
