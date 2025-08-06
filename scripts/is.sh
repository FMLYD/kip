export CUDA_VISIBLE_DEVICES=2

for p in 0.1 0.2 0.3 0.4
do
python exper_standard_kip.py \
  --model KPI \
  --dataset_name ionosphere \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p $p \
  --stop 20 \
  --sigma "0.1,0.3,0.5,0.7,0.9,1,3,5,7" 
done
