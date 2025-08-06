export CUDA_VISIBLE_DEVICES=1

python  benchmark_kpi.py \
  --model KPI\
  --dataset_name connectionist_bench_vowel \
  --outpath './results' \
  --lr 0.005 \
  --seed 2024 \
  --epochs 50 \
  --batch_size 128 \
  --p 0.1 --stop 30 \
  --sigma "0.1,0.5,1,3,5,7,9"\
    --k 2\
  --metric nan_manhattan \
  --weights distance


python  benchmark_kpi.py \
  --model KPI\
  --dataset_name connectionist_bench_vowel \
  --outpath './results' \
  --lr 0.005 \
  --seed 2024 \
  --epochs 50 \
  --batch_size 128 \
  --p 0.1 --stop 30 \
  --sigma "0.1,0.5,1,3,5,7,9"\
    --k 2\
  --metric nan_manhattan \
  --weights distance


python  benchmark_kpi.py \
  --model KPI\
  --dataset_name connectionist_bench_vowel \
  --outpath './results' \
  --lr 0.005 \
  --seed 2024 \
  --epochs 50 \
  --batch_size 128 \
  --p 0.1 --stop 30 \
  --sigma "0.1,0.5,1,3,5,7,9"\
    --k 6\
  --metric nan_manhattan \
  --weights distance


python  benchmark_kpi.py \
  --model KPI\
  --dataset_name connectionist_bench_vowel \
  --outpath './results' \
  --lr 0.005 \
  --seed 2024 \
  --epochs 50 \
  --batch_size 128 \
  --p 0.1 --stop 30 \
  --sigma "0.1,0.5,1,3,5,7,9"\
    --k 12\
  --metric nan_manhattan \
  --weights distance

