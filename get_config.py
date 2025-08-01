import os
import yaml
from collections import defaultdict

def extract_best_mae_models(base_path="./main_knn"):
    results = defaultdict(lambda: defaultdict(dict))

    for dataset in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset)
        if not os.path.isdir(dataset_path):
            continue
            
        for p_miss in os.listdir(dataset_path):
            p_miss_path = os.path.join(dataset_path, p_miss)
            if not os.path.isdir(p_miss_path):
                continue

            best_mae = float("inf")
            best_info = None

            for filename in os.listdir(p_miss_path):
                if not filename.endswith(".yaml") or "_MCAR_" not in filename:
                    continue
                # if '-m' in filename:
                #     continue
                filepath = os.path.join(p_miss_path, filename)

                try:
                    with open(filepath, "r") as f:
                        data = yaml.safe_load(f)
                        mae = data.get("MAE", None)
                        wass=data.get('WASS',None)
                        if mae is None:
                            continue
                except Exception as e:
                    print(f"Failed to read {filepath}: {e}")
                    continue

                if mae < best_mae:
                    best_mae = mae
                    # filename format: {model}_MCAR_{k}.yaml
                    model_k = filename.replace(".yaml", "")
                    parts = model_k.split("_MCAR_")
                    if len(parts) == 2:
                        model = parts[0]
                        try:
                            k = int(parts[1])
                            best_info = {"model": model, "k": k,"mae":best_mae,'wass':wass}
                        except ValueError:
                            continue
            
            if best_info:
                results[dataset][float(p_miss)] = best_info

    return results
best_models=extract_best_mae_models()
import json

with open("best_models.json", "w") as f:
    json.dump(best_models, f, indent=4)
