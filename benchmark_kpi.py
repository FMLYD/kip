import gc

import torch.cuda

from model.ot_imp import OTImputation
import sys
import warnings
import pandas as pd
import numpy as np
import os
import argparse
from hyperimpute.plugins.imputers import Imputers, ImputerPlugin
from utils.utils import RMSE, MAE,MSE
from hyperimpute.plugins.utils.simulate import simulate_nan
from hyperimpute.utils.optimizer import EarlyStoppingExceeded, create_study
import hyperimpute.plugins.core.params as params
from hyperimpute.plugins.imputers.plugin_EM import EMPlugin
from hyperimpute.plugins.imputers.plugin_gain import GainPlugin
from hyperimpute.plugins.imputers.plugin_ice import IterativeChainedEquationsPlugin
from hyperimpute.plugins.imputers.plugin_median import MedianPlugin
from hyperimpute.plugins.imputers.plugin_mean import MeanPlugin
from hyperimpute.plugins.imputers.plugin_mice import MicePlugin
from hyperimpute.plugins.imputers.plugin_miracle import MiraclePlugin
from hyperimpute.plugins.imputers.plugin_missforest import MissForestPlugin
from hyperimpute.plugins.imputers.plugin_miwae import MIWAEPlugin
from hyperimpute.plugins.imputers.plugin_softimpute import SoftImpute
from hyperimpute.plugins.imputers.plugin_most_frequent import MostFrequentPlugin
import ot
from utils.utils import enable_reproducible_results, kip_simulate_scenarios
from sklearn.preprocessing import MinMaxScaler

from model.ot_imp import *
from utils.model_others import KNNImputation, MissForestImputation
from dataloaders import dataset_loader
import torch
from model.CSDITModel import CSDITImputation
from model.MissDiff import MissDiffImputation
from model.remasker.remasker_impute import ReMasker
import yaml

torch.set_default_tensor_type('torch.FloatTensor')

parser = argparse.ArgumentParser(prog='Basic')
parser.add_argument('--model', default='miwae')
parser.add_argument('--feature_num', default=0, type=int)
parser.add_argument('--seed', default=2025, type=int)
# parser.add_argument('--data', default='baicheng')
parser.add_argument('--outpath', default='./results/')
parser.add_argument('--verbose', default=1)
parser.add_argument('--dataset_name', default="blood_transfusion", type=str)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--sigma',default='0.01,0.1,1,3,5,7,9,100,1000')
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--k', default=10, type=int)
parser.add_argument('--stop', default=30, type=int)

parser.add_argument('--loss',default='mae')
parser.add_argument('--p', default=0.1, type=float)
parser.add_argument('--weights',default='mae')
# parse_args operations
from get_config import best_models
def nan_manhattan(x, y, **kwargs):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if not np.any(mask):
        return np.inf
    return np.sum(np.abs(x[mask] - y[mask])) / np.sum(mask)

args = parser.parse_args()
k, model=best_models[args.dataset_name][args.p]['k'],best_models[args.dataset_name][args.p]['model']
weights={'knn-u':'uniform','knn':'distance','knn-m-u':'uniform','knn-m-d':'distance'}[model]
metric={'knn-u':'nan_euclidean','knn':'nan_euclidean','knn-m-u':nan_manhattan,'knn-m-d':nan_manhattan}[model]
print(k,weights,metric)

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
args.sigma=args.sigma.split(',')
args.sigma=[float(x) for x in args.sigma]


# warnings.simplefilter("ignore")
models = {
                'multilaplaciankip':multilaplacianKIPImputation(batch_size=512, lr=5e-3, n_epochs=50, n_pairs=2, noise=1e-4, labda=1.0, normalize=1, initializer=KNNImputation(k=5, weights="distance"), replace=False,sigma=args.sigma),
                'adapkip':AdaptiveMultiKIPImputation(batch_size=512, lr=5e-3, n_epochs=50, n_pairs=2, noise=1e-4, labda=1.0, normalize=1, initializer=KNNImputation(k=5, weights="distance"), replace=False,sigma=args.sigma),

            'KPI':multiKIPImputation(batch_size=args.batch_size, lr=args.lr, n_epochs=args.epochs, n_pairs=2, noise=1e-4, labda=1.0, normalize=1, initializer=KNNImputation(k=k, weights=weights,metric=metric), replace=False,sigma=args.sigma,loss=args.loss,stop=args.stop),
            'multilinearkip':multiLinearKIPImputation(batch_size=512, lr=5e-3, n_epochs=50, n_pairs=2, noise=1e-4, labda=1.0, normalize=1, initializer=KNNImputation(k=5, weights="distance"), replace=False,sigma=args.sigma),
            'multipolykip':multipolyKIPImputation(batch_size=512, lr=5e-3, n_epochs=50, n_pairs=2, noise=1e-4, labda=1.0, normalize=1, initializer=KNNImputation(k=5, weights="distance"), replace=False),

        'kip':KIPImputation(batch_size=512, lr=args.lr, n_epochs=50, n_pairs=2, noise=1e-4, labda=1.0, normalize=1, initializer=KNNImputation(k=5, weights="distance"), replace=False),
        'mean': MeanPlugin(),
        'median': MedianPlugin(),
        'mostfrequent': MostFrequentPlugin(),
        'gain': GainPlugin(batch_size=512, hint_rate=0.8, loss_alpha=10, n_epochs=400),
        'em': EMPlugin(maxit=500, convergence_threshold=1e-12),
        'mice': MicePlugin(n_imputations=1, max_iter=100, ),
        'miracle': MiraclePlugin(batch_size=512, lr=1e-3, max_steps=50, n_hidden=32, reg_beta=1, reg_lambda=0.1,
                                 seed_imputation='median', window=10),
        'si': SoftImpute(maxit=1000, convergence_threshold=1e-5, max_rank=2, shrink_lambda=0, cv_len=3,
                         random_state=0, ),
        # 'ice': IterativeChainedEquationsPlugin(max_iter=500), # it is implemented by hyperimpute. Use the sklearn implementation instead.
        'ice': MicePlugin(n_imputations=1, max_iter=100, ),
        # 'missforest': MissForestPlugin(max_iter=500),  # it is implemented by hyperimpute. Use the sklearn implementation instead.
        'missforest': MissForestImputation(n_trees=10, max_depth=2, min_samples_split=2),
        'miwae': MIWAEPlugin(K=1, batch_size=512, latent_size=16, n_epochs=300, n_hidden=32),
        'lapot': OTLapImputation(batch_size=512, lr=1e-2, n_epochs=100, n_pairs=4, noise=1e-4, numItermax=1000,
                                 numItermaxInner=1000, reg_eta=1, reg_sim='knn', reg_simparam=7, stopThr=1e-3,
                                 stopThrInner=1e-3),
        'lapot2': OTLapImputation(batch_size=512, lr=1e-2, n_epochs=100, n_pairs=4, noise=1e-4, numItermax=1000,
                                  numItermaxInner=1000, reg_eta=1, reg_sim='knn', reg_simparam=7, stopThr=1e-3,
                                  stopThrInner=1e-3, normalize=0),
        'tdm': TDMImputation(batch_size=512, lr=1e-2, n_epochs=100, noise=1e-4, numItermax=1000, stopThr=1e-3,
                             normalize=1, reg_sk=0.01, net_hidden=16, net_indim=ground_truth.shape[1], net_depth=2),
        'otrr': OTRRimputation(n_pairs=10, lr=1e-3, reg_sk=0.1, batch_size=128),
        'sink': OTImputation(batch_size=512, lr=1e-2, n_epochs=100, n_pairs=4, noise=1e-4, numItermax=1000,
                             stopThr=1e-3, normalize=0),
        'knn': KNNImputation(k=args.k, weights="distance"),  # 1-3
        'knn-u': KNNImputation(k=args.k, weights="uniform"),  # 1-3
        'CSDIT': CSDITImputation(layer_number=2, n_channels=16, side_dim=32,
                 particle_number=50,
                 diff_embedding=64, heads_num=2,
                 batch_size=512,
                 epochs=200,
                 lr=1.0e-3,
                 diff_steps=100, schedule="quad",
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
        'MissDiff': MissDiffImputation(layer_number=2, n_channels=16, side_dim=32,
                 particle_number=50,
                 diff_embedding=128, heads_num=2,
                 batch_size=512,
                 epochs=200,
                 lr=1.0e-3,
                 diff_steps=100, schedule="quad",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        # 'remasker': ReMasker()

    }



SCENARIO = [
    # "MAR",
    # "MNAR",
    "MCAR"
]

P_MISS = [0.1,0.2,0.3,0.4]
feature_drop=[0.1,0.3,0.5,0.7,0.9]
enable_reproducible_results(args.seed)
X = ground_truth
diff_model_list = ["CSDI_T", "MissDiff(VP)", "MissDiff(VE)"]
# diff_logic = True if args.model in diff_model_list else False
diff_logic = False

imputation_scenarios = kip_simulate_scenarios(X,  diff_model=diff_logic, mechanisms=SCENARIO, percentages=P_MISS,feature_drop=feature_drop)

print(f"[Info] We are running model: {args.model}")

results = []
result_df = pd.DataFrame()
print(SCENARIO, P_MISS)
def load_data(file_path):
    import yaml
    try:
        with open(file_path, encoding='utf-8') as file:
            return yaml.load(file.read(), Loader=yaml.FullLoader)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {'WASS':9999999999999}
for scenario in SCENARIO:
    for p_miss in P_MISS:
        for model_name in args.model.split(','):
            print(model_name)
            enable_reproducible_results(args.seed)
            # k, model=best_models[args.dataset_name][p_miss]['k'],best_models[args.dataset_name][p_miss]['model']
            # weights={'knn-u':'uniform','knn':'distance','knn-m-u':'uniform','knn-m-d':'distance'}[model]
            # metric={'knn-u':'nan_euclidean','knn':'nan_euclidean','knn-m-u':nan_manhattan,'knn-m-d':nan_manhattan}[model]
            x, x_miss, mask = imputation_scenarios[scenario][p_miss]
            model = models[model_name]
            # print(k,weights,metric)
        
            # if model_name in ['lapot', 'tdm', 'otrr', 'lapot2', 'knn', 'knn-l', 'missforest', 'si', 'sink']:
            if model_name not in ['mean', 'median', 'mostfrequent', 'gain', 'em', 'mice', 'miracle', 'ice', 'mice',
                                'miwae']:
                if model_name in ['multikip']:
                    model.p_miss=p_miss
                    x_impute = model.fit_transform(x_miss.copy().values,x.copy().values)
                else:
                    model.p_miss=p_miss
                    x_impute = model.fit_transform(x_miss.copy().values)
            else:
                model._fit(x_miss.copy())
                x_impute = model._transform(x_miss.copy())
            if type(x_impute) is pd.DataFrame:
                x_impute = x_impute.values

            rmse = RMSE(x_impute, x.values, mask.values)
            mae = MAE(x_impute, x.values, mask.values)
            # M = mask.sum(1) > 0
            mse=MSE(x_impute, x.values, mask.values)
            dist = ot.dist(x_impute, x.values, metric='sqeuclidean', p=2)
            M = mask.sum(1) > 0
            nimp = M.sum().item()
            dists = ((x_impute[M][:, None] - x.values[M]) ** 2).sum(2) / 2.
            wass = ot.emd2(np.ones(nimp) / nimp, np.ones(nimp) / nimp, dists)
            
            result={"RMSE": rmse.item(), "MAE": mae.item(), "WASS": wass,"MSE":mse.item()}
            if model_name in ['kip','multikip','multilaplaciankip','multilinearkip','multipolykip']:
                data=load_data(f'./main_results/{args.dataset_name}/{model_name}_{scenario}_{p_miss}.yaml')
                try:
                    old_mae=data['MAE']
                    if mae.item()<old_mae:
                        
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

            # if args.verbose == 0:
            #     results.append(
            #         f"{model_name},{scenario},{p_miss},{args.seed},{round(rmse, 5)},{round(mae, 5)},{round(wass, 5)}\n")
            # else:
            #     rmses = '|'.join(RMSE(x_impute, x.values, mask.values, verbose=1))
            #     maes = '|'.join(MAE(x_impute, x.values, mask.values, verbose=1))
            #     results.append(
            #         f"{model_name},{scenario},{p_miss},{args.seed},{round(rmse, 5)},{round(mae, 5)},{round(wass, 5)},{rmses},{maes}\n")

# [print(item) for item in results]
# os.makedirs(args.outpath) if not os.path.exists(args.outpath) else None
csv_name = (f"model_{args.model}_data_{args.dataset_name}_seed_{args.seed}.csv")
# result_df.to_csv(os.path.join(args.outpath, csv_name), index=None)

# print(result_df)
# print(result_df.mean(axis=0))