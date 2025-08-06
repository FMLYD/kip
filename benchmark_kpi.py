import gc

import torch.cuda

from model.ot_imp import OTImputation
import pandas as pd
import numpy as np
import os
import argparse
from utils.utils import RMSE, MAE,MSE
from model.KPI import *
import ot
from utils.utils import enable_reproducible_results, kip_simulate_scenarios

from model.ot_imp import *
from utils.model_others import KNNImputation
from dataloaders import dataset_loader
import torch
import yaml

def load_data(file_path):
    try:
        with open(file_path, encoding='utf-8') as file:
            return yaml.load(file.read(), Loader=yaml.FullLoader)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {'WASS':9999999999999}

torch.set_default_tensor_type('torch.FloatTensor')

parser = argparse.ArgumentParser(prog='Basic')
parser.add_argument('--model', default='miwae')
parser.add_argument('--feature_num', default=0, type=int)
parser.add_argument('--seed', default=2025, type=int)
parser.add_argument('--outpath', default='./results/')
parser.add_argument('--verbose', default=1)
parser.add_argument('--dataset_name', default="blood_transfusion", type=str)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--sigma',default='0.01,0.1,1,3,5,7,9,100,1000')
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--k', default=10, type=int)
parser.add_argument('--stop', default=30, type=int)

parser.add_argument('--loss',default='mse')
parser.add_argument('--p', default=0.1, type=float)
parser.add_argument('--weights',default='mae')
# parse_args operations

def nan_manhattan(x, y, **kwargs):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if not np.any(mask):
        return np.inf
    return np.sum(np.abs(x[mask] - y[mask])) / np.sum(mask)

args = parser.parse_args()
k, model=best_models[args.dataset_name][args.p]['k'],best_models[args.dataset_name][args.p]['model']
weights = {'knn-u':'uniform','knn':'distance','knn-m-u':'uniform','knn-m-d':'distance'}[model]
metric = {'knn-u':'nan_euclidean','knn':'nan_euclidean','knn-m-u':nan_manhattan,'knn-m-d':nan_manhattan}[model]
print(k, weights, metric)

print(args)
# get the dataset
if not os.path.exists("./datasets"):
    os.makedirs("./datasets")
if not os.path.exists(f"./main_results/{args.dataset_name}"):
    os.makedirs(f"./main_results/{args.dataset_name}")
if not os.path.exists(f"./main_config/{args.dataset_name}"):
    os.makedirs(f"./main_config/{args.dataset_name}")
ground_truth = dataset_loader(args.dataset_name)
args.feature_num = ground_truth.shape[1]

args = parser.parse_args()
args.outpath = f"./{args.outpath}/{args.model}"
args.sigma = args.sigma.split(',')
args.sigma = [float(x) for x in args.sigma]

models = {'KPI': multiKIPImputation(batch_size=args.batch_size, lr=args.lr, n_epochs=args.epochs, n_pairs=2, noise=1e-4, labda=1.0, normalize=1, initializer = KNNImputation(k=k, weights=weights,metric=metric), replace=False,sigma=args.sigma,loss=args.loss,stop=args.stop),}
SCENARIO = ["MCAR"]
P_MISS = [0.1,0.2,0.3,0.4]
feature_drop = [0.1,0.3,0.5,0.7,0.9]
enable_reproducible_results(args.seed)
X = ground_truth
diff_model_list = ["CSDI_T", "MissDiff(VP)", "MissDiff(VE)"]
diff_logic = False

imputation_scenarios = kip_simulate_scenarios(X,  diff_model=diff_logic, mechanisms=SCENARIO, percentages=P_MISS, feature_drop=feature_drop)

print(f"[Info] We are running model: {args.model}")

results = []
result_df = pd.DataFrame()
print(SCENARIO, P_MISS)

for scenario in SCENARIO:
    for p_miss in P_MISS:
        for model_name in args.model.split(','):
            print(model_name)
            enable_reproducible_results(args.seed)
           
            x, x_miss, mask = imputation_scenarios[scenario][p_miss]
            model = models[model_name]
            model.p_miss = p_miss
            x_impute = model.fit_transform(x_miss.copy().values,x.copy().values)
                
            if type(x_impute) is pd.DataFrame:
                x_impute = x_impute.values

            rmse = RMSE(x_impute, x.values, mask.values)
            mae = MAE(x_impute, x.values, mask.values)
            mse = MSE(x_impute, x.values, mask.values)
            dist = ot.dist(x_impute, x.values, metric='sqeuclidean', p=2)
            M = mask.sum(1) > 0
            nimp = M.sum().item()
            dists = ((x_impute[M][:, None] - x.values[M]) ** 2).sum(2) / 2.
            wass = ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, dists)
            
            result={"RMSE": rmse.item(), "MAE": mae.item(), "WASS": wass, "MSE":mse.item()}
            if model_name in ['kip','multikip','multilaplaciankip','multilinearkip','multipolykip']:
                data = load_data(f'./main_results/{args.dataset_name}/{model_name}_{scenario}_{p_miss}.yaml')
                try:
                    old_mae = data['MAE']
                    if mae.item() < old_mae:
                        with open(f'./main_results/{args.dataset_name}/{model_name}_{scenario}_{p_miss}.yaml','w') as file:
                            yaml.dump(result,file)
                        with open(f'./main_config/{args.dataset_name}/{model_name}_{scenario}_{p_miss}.yaml','w') as file:
                            yaml.dump(args,file)
                except:
                    with open(f'./main_results/{args.dataset_name}/{model_name}_{scenario}_{p_miss}.yaml','w') as file:
                            yaml.dump(result,file)
                    with open(f'./main_config/{args.dataset_name}/{model_name}_{scenario}_{p_miss}.yaml','w') as file:
                        yaml.dump(args,file)
            else:
                with open(f'./main_results/{args.dataset_name}/{model_name}_{scenario}_{p_miss}.yaml','w') as file:
                    
                    yaml.dump(result,file)
            result_dict = {"model_name": model_name, "missing": scenario, "p_miss": p_miss,
                        "seed": args.seed, "rmse": rmse, "mae": mae, "wass": wass,"mse":mse}

            result_df = pd.concat([result_df, pd.DataFrame(result_dict, index=[0])], axis=0)
            print(result)
            del model, x, x_miss, mask, x_impute
            gc.collect()
            torch.cuda.empty_cache()
            
csv_name = (f"model_{args.model}_data_{args.dataset_name}_seed_{args.seed}.csv")

