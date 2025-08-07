export CUDA_VISIBLE_DEVICES=2
python  benchmark_kpi.py \
  --model KPI \
  --dataset_name qsar_biodegradation \
  --outpath './results' \
  --lr 0.001 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.1 \
  --k 2\
  --metric nan_manhattan \
  --weights uniform
  
  python  benchmark_kpi.py \
  --model KPI \
  --dataset_name qsar_biodegradation \
  --outpath './results' \
  --lr 0.001 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.2 \
  --k 3\
  --weights uniform

  
  python  benchmark_kpi.py \
  --model KPI \
  --dataset_name qsar_biodegradation \
  --outpath './results' \
  --lr 0.001 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.3 \
  --k 3\
  --weights uniform
  
  python  benchmark_kpi.py \
  --model KPI \
  --dataset_name qsar_biodegradation \
  --outpath './results' \
  --lr 0.001 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.4 \
  --sigma "0.1,0.3,0.5,0.7,1,3,5" \
  --k 5\
  --metric nan_manhattan \
  --weights uniform
