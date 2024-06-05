#!/bin/bash

####################################################################

#File Name    : multiling_ili.sbatch
#Author       : Niti Mishra
#Email        : niti.mishra@isglobal.org
#Description  : Finetune MLM for ILI detection with best hyperparameters
#Creation Date: 2-21-2024

####################################################################

# Envs
####################################################################
#HPC

#SBATCH --job-name=multiling_ili
#SBATCH --output=logs/%j.out 
#SBATCH --error=logs/%j.err
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=niti.mishra@isglobal.org
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=24gb

####################################################################

# load your required modules
###########################

# source /gaueko0/users/nmishra/niti_venv/niti-transformers4.20/bin/activate      #to run on xirimiri
source /gaueko0/users/nmishra/niti_venv/trumoi-transformers-4.20/bin/activate     #to run on trumoi

# best run
######################################

# export required environment variables
wandb disabled

# train on one to many original + translations
CUDA_LAUNCH_BLOCKING=1 python /gaueko0/users/nmishra/multiling_fludetection/scripts/multilingual_ILI_classification.py \
--data_file /gaueko0/users/nmishra/multiling_fludetection/data/all/fulldata_revisedcateg.csv \
--params_file /gaueko0/users/nmishra/multiling_fludetection/params_revisedcateg.tsv \
--output_dir /gaueko0/users/nmishra/multiling_fludetection/final_evals/eval_revisedcateg \
--temp_model_dir /gscratch3/users/nmishra \
--split_index_filename split_idx_one_to_many \
--language_evaluation True \
--language_evaluation_column eval_lang

CUDA_LAUNCH_BLOCKING=1 python /gaueko0/users/nmishra/multiling_fludetection/scripts/multilingual_ILI_classification.py \
--data_file /gaueko0/users/nmishra/multiling_fludetection/data/all/fulldata_revisedcateg.csv \
--params_file /gaueko0/users/nmishra/multiling_fludetection/params_revisedcateg.tsv \
--output_dir /gaueko0/users/nmishra/multiling_fludetection/final_evals/eval_revisedcateg \
--temp_model_dir /gscratch3/users/nmishra \
--split_index_filename split_idx_one_to_many \
--language_evaluation True \
--language_evaluation_column eval_lang

CUDA_LAUNCH_BLOCKING=1 python /gaueko0/users/nmishra/multiling_fludetection/scripts/multilingual_ILI_classification.py \
--data_file /gaueko0/users/nmishra/multiling_fludetection/data/all/fulldata_revisedcateg.csv \
--params_file /gaueko0/users/nmishra/multiling_fludetection/params_revisedcateg.tsv \
--output_dir /gaueko0/users/nmishra/multiling_fludetection/final_evals/eval_revisedcateg \
--temp_model_dir /gscratch3/users/nmishra \
--split_index_filename split_idx_one_to_many \
--language_evaluation True \
--language_evaluation_column eval_lang

CUDA_LAUNCH_BLOCKING=1 python /gaueko0/users/nmishra/multiling_fludetection/scripts/multilingual_ILI_classification.py \
--data_file /gaueko0/users/nmishra/multiling_fludetection/data/all/fulldata_revisedcateg.csv \
--params_file /gaueko0/users/nmishra/multiling_fludetection/params_revisedcateg.tsv \
--output_dir /gaueko0/users/nmishra/multiling_fludetection/final_evals/eval_revisedcateg \
--temp_model_dir /gscratch3/users/nmishra \
--split_index_filename split_idx_one_to_many \
--language_evaluation True \
--language_evaluation_column eval_lang

CUDA_LAUNCH_BLOCKING=1 python /gaueko0/users/nmishra/multiling_fludetection/scripts/multilingual_ILI_classification.py \
--data_file /gaueko0/users/nmishra/multiling_fludetection/data/all/fulldata_revisedcateg.csv \
--params_file /gaueko0/users/nmishra/multiling_fludetection/params_revisedcateg.tsv \
--output_dir /gaueko0/users/nmishra/multiling_fludetection/final_evals/eval_revisedcateg \
--temp_model_dir /gscratch3/users/nmishra \
--split_index_filename split_idx_one_to_many \
--language_evaluation True \
--language_evaluation_column eval_lang