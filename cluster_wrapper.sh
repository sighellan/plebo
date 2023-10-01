#!/bin/sh

NUM_EXP=50
MAX_PARALLEL_JOBS=30
echo "Number of experiments: ${NUM_EXP}"

export STUDENT_ID=$(whoami)

EXPT_FILE=test_exps.txt


echo "Scheduling experiment file $EXPT_FILE"
sbatch --array=1-${NUM_EXP}%${MAX_PARALLEL_JOBS} cluster_script.sh $EXPT_FILE
echo "Done!"
