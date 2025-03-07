import argparse
import scipy
import tqdm

from data_builder import load_data
from model import load_tokenizer

def draw_figure(x, y, title, xlabel, ylabel, output_file):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(output_file)
    plt.close()
    

def experiment(args):
    # load tokenizer
    tokenizer = load_tokenizer(args.model_name, args.dataset, args.cache_dir)

    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])

    # for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
    for idx in range(10):
    # for idx in [2]:

        # sampled_text = data["original"][idx]
        # tokenized = tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        # labels = tokenized.input_ids[:, 1:]
        # distance = []        
        # for i in range(0, len(labels[0])):
        #     flag = 0
        #     for j in range(i-1, 0, -1):
        #         if labels[0][i] == labels[0][j]:
        #             flag = 1
        #             distance.append(i - j)
        #             break
        #     if flag == 0:
        #         distance.append(i)
        # normal_distance = []
        # for i in range(1, len(labels[0])):
        #     normal_distance.append(distance[i] / i);
        # y_smooth = scipy.signal.savgol_filter(normal_distance, 50, 3)
        # draw_figure(range(1, len(labels[0])), y_smooth, "Distance between tokens", "Token index", "Distance", args.output_file+"_raw-data_"+str(idx))



        sampled_text = data["sampled"][idx]
        tokenized = tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        distance = []        
        for i in range(0, len(labels[0])):
            flag = 0
            for j in range(i-1, 0, -1):
                if labels[0][i] == labels[0][j]:
                    flag = 1
                    distance.append(i - j)
                    break
            if flag == 0:
                distance.append(i)
        normal_distance = []
        for i in range(1, len(labels[0])):
            normal_distance.append(distance[i] / i);
        y_smooth = 1-scipy.signal.savgol_filter(normal_distance, 50, 3)
        
        # FFT
        w_smooth = scipy.fftpack.rfft(y_smooth)
        f = scipy.fftpack.rfftfreq(len(y_smooth), d=1)
        spectrum = w_smooth**2
        draw_figure(f, spectrum, "FFT of distance between tokens", "Frequency", "Power", args.output_file+"_fft_"+str(idx))

        # draw_figure(range(1, len(labels[0])), y_smooth, "Distance between tokens", "Token index", "Distance", args.output_file+"_"+str(idx))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/local-wiki-zh_Qwen2-1_5B")
    # parser.add_argument('--output_file', type=str, default="./exp_test/results/local-wiki-zh")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_main/data/local-wiki-zh_Qwen2-1.5B")
    parser.add_argument('--model_name', type=str, default="Qwen2-0.5B")
    parser.add_argument('--discrepancy_analytic', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
