export CUDA_VISIBLE_DEVICES=2

for model in       multikip

do
for lr in 0.01
do
for seed in   2024
do
for sigma in "1,3,5,7,9" 
do


for epochs in  500
do
for k in  10
do
for bs in  128
do
for dataset in              qsar_biodegradation  
do
for p in 0.1 
do
python  exper_standard_kip.py \
  --model $model \
  --dataset_name $dataset \
  --outpath './results' \
  --lr $lr \
  --seed $seed \
  --epochs $epochs \
  --batch_size $bs \
  --p $p
done
done
done
done
done
done
done
done
done
done
done