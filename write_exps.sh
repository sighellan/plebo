#!/bin/sh

STEPS=50
NUM_SEEDS=1
PROBLEM_ID_SYNTH="2023-08-25-14-43-26"
TASK_STR_SYNTH="-task-"
PROBLEM_ID_SATELLITE="strong"
TASK_STR_SATELLITE=""
MIN_TASK=0
MAX_TASK=49
FILE="test_exps.txt"
STARTING_POINTS=10

rm ${FILE}

for BENCHMARK in "satellite" #"synth"
do
    if [ $BENCHMARK = "synth" ]
    then
        PROBLEM_ID=${PROBLEM_ID_SYNTH}
        TASK_STR=${TASK_STR_SYNTH}
    fi
    if [ $BENCHMARK = "satellite" ]
    then
        PROBLEM_ID=${PROBLEM_ID_SATELLITE}
        TASK_STR=${TASK_STR_SATELLITE}
    fi
    
    for ((task_num=${MIN_TASK}; task_num<=${MAX_TASK}; task_num++))
    do
        for ACQ in "HBO_true" "Shared" "Gamma" "RS" "Initial" "EI" "UCB" "Dir_trans" "BoTorch" "HBO_numpyro"
        do
            echo "python do_BO.py --benchmark=$BENCHMARK --steps=$STEPS --num_seeds=${NUM_SEEDS} --problem_id=${PROBLEM_ID} --task_id=${TASK_STR}${task_num} --acq_name=$ACQ --starting_points=${STARTING_POINTS}" >> ${FILE}
        done
    done
done
