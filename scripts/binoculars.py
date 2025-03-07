import argparse
import tqdm
import json
import numpy as np
from data_builder import load_data
from metrics import get_roc_metrics, get_precision_recall_metrics

from binoculars import Binoculars
from binoculars.detector import BINOCULARS_ACCURACY_THRESHOLD as THRESHOLD


bino = Binoculars()


def experiment(args):
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    # eval criterions
    name = "binoculars"
    criterion_fn = bino.compute_score

    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        # original text
        original_crit = 1 - criterion_fn(original_text)
        # sampled text
        sampled_crit = 1 - criterion_fn(sampled_text)
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})
        
    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                     'samples': [x["sampled_crit"] for x in results]}
    
    # predictions = {'real': np.where(np.array([x["original_crit"] for x in results]) > THRESHOLD, 1, 0),
    #               'samples': np.where(np.array([x["sampled_crit"] for x in results]) > THRESHOLD, 1, 0)}
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    
    # results
    results_file = f'{args.output_file}.{name}.json'
    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': results,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/xsum_gpt2")

    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_test/data/xsum_gpt2")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)