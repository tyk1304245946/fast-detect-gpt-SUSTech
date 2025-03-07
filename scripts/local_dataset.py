from datasets import Dataset, load_dataset

import os
import json

def process_original(dataset):
    dataset['original'] = dataset['input'] + dataset['output']
    return dataset

def process_sampled(dataset):
    dataset['sampled'] = dataset['input'] + dataset['output']
    return dataset

def save_data(output_file, args, data):
    # # write args to file
    # args_file = f"{output_file}.args.json"
    # with open(args_file, "w") as fout:
    #     json.dump(args.__dict__, fout, indent=4)
    #     print(f"Args written into {args_file}")

    # write the data to a json file in the save folder
    data_file = f"{output_file}.raw_data.json"
    with open(data_file, "w") as fout:
        json.dump(data, fout, indent=4)
        print(f"Raw data written into {data_file}")


datasets = ['news-zh', 'webnovel', 'wiki-zh']
# datasets = ['wiki-zh']


# generate_model = ['qwen2-1_5b-base', 'qwen2-7b-base', 'qwen2-72b-base']
# generate_model = ['qwen2-7b-base']

generate_model = ['yi1_5-6b','yi1_5-9b', 'yi1_5-34b','qwen2-1_5b-base', 'qwen2-7b-base', 'qwen2-72b-base']

model_names = {
    'qwen2-1_5b-base': 'Qwen2-1.5B',
    'qwen2-7b-base': 'Qwen2-7B',
    'qwen2-72b-base': 'Qwen2-72B',
    'yi1_5-6b': 'Yi-1.5-6B',
    'yi1_5-9b': 'Yi-1.5-9B',
    'yi1_5-34b': 'Yi-1.5-34B',
}

# model_names = {
#     'yi1_5-6b': 'Yi-1.5-6B',
#     'yi1_5-9b': 'Yi-1.5-9B',
#     'yi1_5-34b': 'Yi-1.5-34B',
# }


for dataset in datasets:

    # src_dataset_1 = Dataset.from_file("/nas/cxsj_data/qwen2-0_5b-base-inferenced.human." + dataset + "/data-00000-of-00001.arrow")
    # src_dataset_1 = src_dataset_1.remove_columns(['logprobs'])

    src_dataset_1 = Dataset.from_file("/nas/cxsj_data/qwen2-0_5b-base-inferenced.human." + dataset + "/data-00000-of-00001.arrow")
    src_dataset_1 = src_dataset_1.remove_columns(['logprobs'])

    src_dataset_1 = src_dataset_1.map(process_original)

    for model in generate_model:
        src_dataset_2 = Dataset.from_file("/nas/cxsj_data/qwen2-0_5b-base-inferenced." + model + "-generated." + dataset + "/data-00000-of-00001.arrow")
        print(src_dataset_2)

        src_dataset_2 = src_dataset_2.remove_columns(['logprobs'])
        src_dataset_2 = src_dataset_2.map(process_sampled)

        data = {
            "original": [],
            "sampled": [],
        }

        data["original"] = src_dataset_1["original"]
        data["sampled"] = src_dataset_2["sampled"]

        # for i in range(len(data['original'])):
        for i in range(10):
            if len(data['original'][i])<100 or len(data['sampled'][i])<100:
                del data['original'][i];
                del data['sampled'][i];


        model_name = model_names[model]


        output_file = "./exp_main/data/local-" + dataset + "_" + model_name
        save_data(output_file, None, data)

# dataset2 = Dataset.from_file("/nas/cxsj_data/qwen2-0_5b-base-inferenced.qwen2-1_5b-base-generated.news-zh/data-00000-of-00001.arrow")

# print(dataset2)
# print(dataset2[0])



# load /home/cxsj24f-g1/fast-detect-gpt-SUSTech/exp_main/data
# dataset3 = Dataset.load_from_disk("/home/cxsj24f-g1/fast-detect-gpt-SUSTech/exp_main/data/local-news-zh_Qwen2-1.5B.raw_data.json")
# dataset3 = load_dataset('json', data_files="/home/cxsj24f-g1/fast-detect-gpt-SUSTech/exp_main/data/local-news-zh_Qwen2-1.5B.raw_data.json")
# print(dataset3)

# # dataset4 = dataset3 = Dataset.load_from_disk("/home/cxsj24f-g1/fast-detect-gpt-SUSTech/exp_main/data/THUCNews_Qwen2-1.5B.t5-3b.perturbation_100.raw_data.json")
# dataset4 = load_dataset('json', data_files="/home/cxsj24f-g1/fast-detect-gpt-SUSTech/exp_main/data/THUCNews_Qwen2-1.5B.raw_data.json")
# print(dataset4)


# print(dataset3['train'][0])
# print(dataset4['train'][0])