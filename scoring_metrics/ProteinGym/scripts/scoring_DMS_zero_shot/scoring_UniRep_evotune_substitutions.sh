#!/bin/bash 

source ../zero_shot_config.sh
source activate protein_fitness_prediction_hsu

export OMP_NUM_THREADS=1

export model_path="path to folder containing evotuned UniRep models"
export output_dir=${DMS_output_score_folder_subs}/UniRep_evotuned
export DMS_index="Experiment index to run (E.g. 0,1,2,...,217)"

python ../../proteingym/baselines/unirep/unirep_inference.py \
            --model_path $model_path \
            --data_path $DMS_data_folder_subs \
            --output_dir $output_dir \
            --mapping_path $DMS_reference_file_path_subs \
            --DMS_index $DMS_index \
            --batch_size 32 \
            --evotune