import argparse

from data_builder import load_data

def experiment(args):
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])

    for i in range(10):
        print(str(i) +'-1')
        print(data['original'][i])
        print(str(i) +'-2')
        print(data['sampled'][i])
        print("\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/local-news-zh_Qwen2-7B")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_main/data/local-news-zh_Qwen2-7B")
    parser.add_argument('--model_name', type=str, default="Qwen2-0.5B")
    parser.add_argument('--discrepancy_analytic', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)