#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
set -e  # exit if error

# prepare folders
exp_path=exp_main
data_path=$exp_path/data
res_path=$exp_path/result2
mkdir -p $exp_path $data_path $res_path


datasets="local-news-zh local-webnovel local-wiki-zh"
# source_models="Qwen2-1.5B Qwen2-7B Qwen2-72B"
source_models="Yi-1.5-6B Yi-1.5-9B Yi-1.5-34B"

echo `date`, Evaluate models in the black-box setting:
# scoring_models="Qwen2-1.5B Qwen2-7B"
# scoring_models="Yi-1.5-6B Yi-1.5-9B Yi-1.5-34B"
scoring_models="Yi-1.5-9B Yi-1.5-34B"

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


# # evaluate DNA-GPT
# for D in $datasets; do
#   for M in $source_models; do
#     echo `date`, Evaluating DNA-GPT on ${D}_${M} ...
#     python scripts/dna_gpt.py --base_model_name $M --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M} \
#                           --cache_dir $HF_HOME/hub/
#   done
# done

# todo: DetectGPT need to adapt Chinese
# # evaluate DetectGPT and its improvement DetectLLM
# for D in $datasets; do
#   for M in $source_models; do
#     echo `date`, Evaluating DetectGPT on ${D}_${M} ...
#     python scripts/detect_gpt.py --scoring_model_name $M --mask_filling_model_name t5-3b --n_perturbations 100 --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M} \
#                           --cache_dir $HF_HOME/hub/
#      # we leverage DetectGPT to generate the perturbations
#     echo `date`, Evaluating DetectLLM methods on ${D}_${M} ...
#     python scripts/detect_llm.py --scoring_model_name $M --dataset $D \
#                           --dataset_file $data_path/${D}_${M}.t5-3b.perturbation_100 --output_file $res_path/${D}_${M} \
#                           --cache_dir $HF_HOME/hub/
#   done
# done


# # Black-box Setting
# echo `date`, Evaluate models in the black-box setting:
# scoring_models="gpt-neo-2.7B"

# # evaluate Fast-DetectGPT
# for D in $datasets; do
#   for M in $source_models; do
#     M1=gpt-j-6B  # sampling model
#     for M2 in $scoring_models; do
#       echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M1}_${M2} ...
#       python scripts/fast_detect_gpt.py --reference_model_name ${M1} --scoring_model_name ${M2} --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2} \
#                           --cache_dir $HF_HOME/hub/
#     done
#   done
# done

# # evaluate DetectGPT and its improvement DetectLLM
# for D in $datasets; do
#   for M in $source_models; do
#     M1=t5-3b  # perturbation model
#     for M2 in $scoring_models; do
#       echo `date`, Evaluating DetectGPT on ${D}_${M}.${M1}_${M2} ...
#       python scripts/detect_gpt.py --mask_filling_model_name ${M1} --scoring_model_name ${M2} --n_perturbations 100 --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2} \
#                           --cache_dir $HF_HOME/hub/
#       # we leverage DetectGPT to generate the perturbations
#       echo `date`, Evaluating DetectLLM methods on ${D}_${M}.${M1}_${M2} ...
#       python scripts/detect_llm.py --scoring_model_name ${M2} --dataset $D \
#                           --dataset_file $data_path/${D}_${M}.${M1}.perturbation_100 --output_file $res_path/${D}_${M}.${M1}_${M2} \
#                           --cache_dir $HF_HOME/hub/
#     done
#   done
# done