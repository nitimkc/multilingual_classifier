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

# # grid search
# ######################################

# # export required environment variables
# wandb sync --no-include-online --clean-old-hours 2

# # train for gridsearch
# CUDA_LAUNCH_BLOCKING=1 python /gaueko0/users/nmishra/multiling_fludetection/scripts/multilingual_ILI_classification_wandb.py \
# --data_file /gaueko0/users/nmishra/multiling_fludetection/data/all/alldata_revised.csv \
# --params_file /gaueko0/users/nmishra/multiling_fludetection/params.json \
# --config_file /gaueko0/users/nmishra/multiling_fludetection/sweep_config.yml \
# --output_dir /gaueko0/users/nmishra/multiling_fludetection/evals/gridsearch_revisedcateg \
# --wandb_tmpdir /gscratch3/users/nmishra



# # best run
# ######################################

# # export required environment variables
# wandb disabled

# Experiment 1, 2 and 5 : train on all + each original tweets (6 trainings)
# ############################################################################
# CUDA_LAUNCH_BLOCKING=1 python /gaueko0/users/nmishra/multiling_fludetection/scripts/multilingual_ILI_classification.py \
# --data_file /gaueko0/users/nmishra/multiling_fludetection/data/all/fulldata_revisedcateg.csv \
# --params_file /gaueko0/users/nmishra/multiling_fludetection/params_revisedcateg.tsv \
# --output_dir /gaueko0/users/nmishra/multiling_fludetection/final_evals/eval_revisedcateg \
# --temp_model_dir /gscratch3/users/nmishra \
# --split_index_filename split_idx_original \
# --language_evaluation True \
# --language_evaluation_column eval_lang

# Experiment 1.1 : train on es with .25, .50 and .75 split of data (3 trainings)
# ##############################################################################
# CUDA_LAUNCH_BLOCKING=1 python /gaueko0/users/nmishra/multiling_fludetection/scripts/multilingual_ILI_classification_learningcurve.py \
# --data_file /gaueko0/users/nmishra/multiling_fludetection/data/all/fulldata_revisedcateg.csv \
# --params_file /gaueko0/users/nmishra/multiling_fludetection/params_revisedcateg.tsv \
# --output_dir /gaueko0/users/nmishra/multiling_fludetection/final_evals/eval_revisedcateg \
# --temp_model_dir /gscratch3/users/nmishra \
# --split_index_filename split_idx_es_learningcurve \
# --language_evaluation True \
# --language_evaluation_column eval_lang

# Experiment 3 : train on each translated to all other (5 trainings)
# ############################################################################
# CUDA_LAUNCH_BLOCKING=1 python /gaueko0/users/nmishra/multiling_fludetection/scripts/multilingual_ILI_classification.py \
# --data_file /gaueko0/users/nmishra/multiling_fludetection/data/all/fulldata_revisedcateg.csv \
# --params_file /gaueko0/users/nmishra/multiling_fludetection/params_revisedcateg.tsv \
# --output_dir /gaueko0/users/nmishra/multiling_fludetection/final_evals/eval_revisedcateg \
# --temp_model_dir /gscratch3/users/nmishra \
# --split_index_filename split_idx_one_to_many \
# --language_evaluation True \
# --language_evaluation_column eval_lang

# Experiment 3.1 : train on each translated to all other (5 trainings) but w/
#                  number of samples reduced to match that of English data 
# TBD
# ############################################################################
# CUDA_LAUNCH_BLOCKING=1 python /gaueko0/users/nmishra/multiling_fludetection/scripts/multilingual_ILI_classification.py \
# --data_file /gaueko0/users/nmishra/multiling_fludetection/data/all/fulldata_revisedcateg.csv \
# --params_file /gaueko0/users/nmishra/multiling_fludetection/params_revisedcateg.tsv \
# --output_dir /gaueko0/users/nmishra/multiling_fludetection/final_evals/eval_revisedcateg \
# --temp_model_dir /gscratch3/users/nmishra \
# --split_index_filename split_idx_one_to_many_reduced \
# --language_evaluation True \
# --language_evaluation_column eval_lang

# Experiment 4 : train on all translated to each other (5 trainings)
# TBD
# ############################################################################
# CUDA_LAUNCH_BLOCKING=1 python /gaueko0/users/nmishra/multiling_fludetection/scripts/multilingual_ILI_classification.py \
# --data_file /gaueko0/users/nmishra/multiling_fludetection/data/all/fulldata_revisedcateg.csv \
# --params_file /gaueko0/users/nmishra/multiling_fludetection/params_revisedcateg.tsv \
# --output_dir /gaueko0/users/nmishra/multiling_fludetection/final_evals/eval_revisedcateg \
# --temp_model_dir /gscratch3/users/nmishra \
# --split_index_filename split_idx_many_to_one \
# --language_evaluation True \
# --language_evaluation_column eval_lang

# Monolingual trainings
# ############################################################################
# CUDA_LAUNCH_BLOCKING=1 python /gaueko0/users/nmishra/multiling_fludetection/scripts/multilingual_ILI_classification.py \
# --data_file /gaueko0/users/nmishra/multiling_fludetection/data/all/fulldata_revisedcateg.csv \
# --params_file /gaueko0/users/nmishra/multiling_fludetection/params_mono_revisedcateg.tsv \
# --output_dir /gaueko0/users/nmishra/multiling_fludetection/evals_mono/eval_revisedcateg \
# --temp_model_dir /gscratch3/users/nmishra \
# --split_index_filename split_idx_original \
# --language_evaluation True \
# --language_evaluation_column eval_lang