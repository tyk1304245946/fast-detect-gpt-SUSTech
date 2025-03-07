# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_main
data_path=$exp_path/data
res_path=$exp_path/result2
mkdir -p $exp_path $data_path $res_path


datasets="local-news-zh local-webnovel local-wiki-zh"
source_models="Qwen2-1.5B Qwen2-7B Qwen2-72B"

echo `date`, Evaluate models in the black-box setting:
scoring_models="Qwen2-0.5B Qwen2-1.5B Qwen2-7B"

# evaluate Fast-DetectGPT
for D in $datasets; do
  for M in $source_models; do
    for M2 in $scoring_models; do
      echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M2}_${M2} ...
      python scripts/fast_detect_gpt.py --reference_model_name ${M2} --scoring_model_name ${M2} --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M2}_${M2} \
                          --cache_dir $HF_HOME/hub/
    done
  done
done