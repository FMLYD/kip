export CUDA_VISIBLE_DEVICES=2

for p in 0.1 
do
python  exper_standard_kip.py \
  --model KPI \
  --dataset_name qsar_biodegradation \
  --outpath './results' \
  --lr 0.01 \
  --seed 2024 \
  --epochs 500 \
  --batch_size 128 \
  --p $p \
  --sigma "0.1,0.3,0.5,0.7,1,3,5" 
done

