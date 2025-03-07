# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib
import matplotlib.pyplot as plt
import argparse
import glob
import json
from os import path
import seaborn as sns
import pandas as pd
import os
import numpy as np

matplotlib.use('Agg')

# plot histogram of sampled on left, and original on right
def save_histogram(predictions, figure_file):
    plt.figure(figsize=(4, 2.5))
    plt.subplot(1, 1, 1)
    plt.hist(predictions["samples"], alpha=0.5, bins='auto', label='Model')
    plt.hist(predictions["real"], alpha=0.5, bins='auto', label='Human')
    plt.xlabel("Sampling Discrepancy")
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(figure_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_files', type=str, default="./exp_test/results/*.json")
    parser.add_argument('--draw', action='store_true')
    parser.add_argument('--heatmap', action='store_true')
    args = parser.parse_args()
    index = ['Qwen2-1.5B', 'Qwen2-7B', 'Qwen2-72B', 'Yi-1.5-6B', 'Yi-1.5-9B', 'Yi-1.5-34B']
    columns = ['Qwen2-0.5B', 'Qwen2-1.5B', 'Qwen2-7B', 'Yi-1.5-6B']
    auc_df0 = pd.DataFrame(index=index, columns=columns, dtype=float)            # local-news-zh
    auc_df1 = pd.DataFrame(index=index, columns=columns, dtype=float)            # local-webnovel
    auc_df2 = pd.DataFrame(index=index, columns=columns, dtype=float)            # local-wiki-zh
    for res_file in glob.glob(args.result_files, recursive=True):
        with open(res_file, 'r') as fin:
            res = json.load(fin)
        if 'metrics' in res:
            n_samples = res['info']['n_samples']
            roc_auc = res['metrics']['roc_auc']
            real = res['predictions']['real']
            samples = res['predictions']['samples']
            if args.heatmap:
                # update auc_df to draw heatmap
                file_name = os.path.basename(res_file)
                category = file_name.split('_')[0]
                models = file_name.split('_')[1].split('B')
                source_model = models[0] + 'B'
                scoring_model = models[1][1:] + 'B'
                match category:
                    case 'local-news-zh' : auc_df0.at[source_model, scoring_model] = roc_auc
                    case 'local-webnovel': auc_df1.at[source_model, scoring_model] = roc_auc
                    case 'local-wiki-zh' : auc_df2.at[source_model, scoring_model] = roc_auc
                    case _ :
                        print("Error: Unknown category out of [local-news-zh, local-webnovel, local-wiki-zh]!")
                        exit
            # print(f"{res_file}: roc_auc={roc_auc:.4f} n_samples={n_samples} r:{np.mean(real):.2f}/{np.std(real):.2f} s:{np.mean(samples):.2f}/{np.std(samples):.2f}")
        else:
            print(f"{res_file}: metrics not found.")
        # draw histogram 
        if args.draw:
            fig_file = f"{res_file}.pdf"
            save_histogram(res['predictions'], fig_file)
            print(f"{fig_file}: histogram figure saved.")

    if args.heatmap:
        seq = [('News',auc_df0), ('Novel', auc_df1), ('Wiki', auc_df2)]
        for (k, df) in seq:
            print("---------------------" + k + "---------------------")
            print(df)
            print()
            sns.set_theme(style="whitegrid")

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for i, (k, df) in enumerate(seq):
                sns.heatmap(df, annot=True, cmap='YlGnBu', fmt=".2f", cbar=True, ax=axes[i], vmin=0.7, vmax=1)
                axes[i].set_title(f'{k} AUC Heatmap')
                axes[i].set_xlabel('Scoring-Model')
                axes[i].set_ylabel('Dataset')
            plt.tight_layout()
            plt.savefig('exp_main/heatmaps/ROC_AUC.png')

            plt.close