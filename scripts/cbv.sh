export CUDA_VISIBLE_DEVICES=1

python  benchmark_kpi.py \
  --model KPI\
  --dataset_name connectionist_bench_vowel \
  --outpath './results' \
  --lr 0.001 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.1 --stop 10 \
    --k 2\
  --metric nan_manhattan \
  --weights distance


python  benchmark_kpi.py \
  --model KPI\
  --dataset_name connectionist_bench_vowel \
  --outpath './results' \
  --lr 0.001 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.2 --stop 10 \
    --k 2\
  --metric nan_manhattan \
  --weights distance


python  benchmark_kpi.py \
  --model KPI\
  --dataset_name connectionist_bench_vowel \
  --outpath './results' \
  --lr 0.001 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.3 --stop 10 \
    --k 6\
  --metric nan_manhattan \
  --weights distance


python  benchmark_kpi.py \
  --model KPI\
  --dataset_name connectionist_bench_vowel \
  --outpath './results' \
  --lr 0.001 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p 0.4 --stop 10 \
    --k 12\
  --metric nan_manhattan \
  --weights distance

