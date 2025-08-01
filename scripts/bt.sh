export CUDA_VISIBLE_DEVICES=2

for p in  0.1 0.2 0.3 0.4
do
python  benchmark_spi.py \
  --model KPI \
  --dataset_name  blood_transfusion  \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p $p\
  --stop 10 \
  --sigma "0.1,0.3,0.5,0.7,1,3,5" 
done
