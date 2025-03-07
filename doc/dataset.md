# 数据集处理

## Source Dataset

数据集目录：`/nas/cxsj_data/`

目前数据集为.arrow格式

```
qwen2-0_5b-base-inferenced.human.news-zh
qwen2-0_5b-base-inferenced.qwen2-1_5b-base-generated.news-zh
qwen2-0_5b-base-inferenced.qwen2-7b-base-generated.news-zh
qwen2-0_5b-base-inferenced.qwen2-72b-base-generated.news-zh
```

数据集包含以下字段：

```
Dataset({
    features: ['input', 'output', 'logprobs'],
    num_rows: 5000
})
```

### load

例: 读取`qwen2-0_5b-base-inferenced.human.news-zh`
```python
from dataset import load_dataset
dataset = Dataset.from_file("/nas/cxsj_data/qwen2-0_5b-base-inferenced.human.news-zh/data-00000-of-00001.arrow")
```

## Target Dataset

存储到项目路径`/home/cxsj24f-g1/fast-detect-gpt-SUSTech/exp_main/data`下

类似已经存在的：

```
THUCNews_Qwen2-1.5B.raw_data.json
```

### 数据集格式
```
DatasetDict({
    train: Dataset({
        features: ['original', 'sampled'],
        num_rows: 500
    })
})
```


## The operation you need to do

只需要先操作`news`部分, wiki和article先不用管

1. 把源数据集中`human`部分的input和output结合在一起，作为输出数据集的`original` feature

2. 将不同模型(`qwen2-1_5b-base`, `qwen2-7b-base`)generate部分的input和output结合在一起，作为输出数据集的`sampled` feature

3. 将数据集按照target dataset的格式保存，并命名为`loacl-news_模型名.raw_data.json`,如`local-news_qwen2-1_5b.raw_data.json`