#!/bin/bash
trap "exit" INT
# ./requirement.sh
train_or_test=${1?Error: Training the model or testing the model}
dataset=${2?Error: Which dataset am I using now?}
version=${3?Error: The version of the experiment}
ckptdownload=${4?Error: Evaluate experiments with the downloaded ckpt or own ckpt}
datadir=${5:-/project_scratch/bo/anomaly_data/}
# datadir=/project_scratch/bo/anomaly_data/
model_type=2d_2d_pure_unet
motion_method=conv3d
if [ $ckptdownload = true ]; then
    expdir=checkpoints/
else
    expdir=/project/bo/exp_data/
fi


if [ $train_or_test = train ]; then
    python3 train_end2end.py --datadir $datadir --expdir $expdir --data_set $dataset --model_type $model_type --motion_method $motion_method --version $version
elif [ $train_or_test = test ]; then
    python3 test_end2end.py --datadir $datadir --expdir $expdir --data_set $dataset --model_type $model_type --motion_method $motion_method --version $version --opt save_score
elif [ $train_or_test = fps ]; then
    python3 end2end_reallife.py --datadir $datadir --expdir $expdir --data_set $dataset --model_type $model_type --motion_method $motion_method --version $version
fi
