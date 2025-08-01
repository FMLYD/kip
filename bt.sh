# seeds yacht_hydrodynamics planning_relax  libras concrete_slump ecoli   climate_model_crashes airfoil_self_noise
# "yeast" "wine_quality_white"  "parkinsons"  qsar_biodegradation blood_transfusion breast_cancer_diagnostic connectionist_bench_vowel ionosphere "parkinsons"  ionosphere connectionist_bench_vowel
export CUDA_VISIBLE_DEVICES=2

for model in       KPI
#  gain ice multikip  lapot mice miracle missforest  miwae si sink tdm  
# CSDIT em gain ice kip knn lapot mice miracle missforest  miwae si sink tdm 
# multilinearkip multipolykip   multilaplaciankip CSDIT em gain ice kip knn lapot mice miracle missforest  miwae si sink tdm 
do
for lr in 0.01
do
for seed in   2024
do
for sigma in "0.1,0.3,0.5,0.7,0.9,1,3,5,7" 
do

# blood_transfusion concrete_compression ionosphere parkinsons qsar_biodegradation  seeds yacht_hydrodynamics planning_relax
# for dataset in        blood_transfusion breast_cancer_diagnostic concrete_compression connectionist_bench_vowel ionosphere parkinsons qsar_biodegradation  wine_quality_white
# 
for epochs in  500
do
for k in  10
do
for bs in  128
do
for dataset in        blood_transfusion 
do
for p in  0.1 0.2 0.3 0.4
do
python  benchmark_spi.py \
  --model $model \
  --dataset_name $dataset \
  --outpath './results' \
  --lr $lr \
  --seed $seed \
  --epochs $epochs \
  --batch_size $bs \
  --p $p\
  --stop 20 
done
done
done
done
done
done
done
done
done
