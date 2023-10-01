#!/bin/sh
#SBATCH -N 1                # Nodes requested
#SBATCH -n 1                # Tasks requested
#SBATCH --cpus-per-task=8   # CPUs requested
#SBATCH --mem=14000         # Memory in Mb
#SBATCH --time=0-12:00:00    # Time required
#SBATCH -o ./Report/slurm-%A_%a.out

export STUDENT_ID=$(whoami)

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

JOBTASK="${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}
export RESDIR=/disk/scratch/${STUDENT_ID}/${JOBTASK}

DATADIR=/home/${STUDENT_ID}/jam/problems/
SAMPLESDIR=/home/${STUDENT_ID}/jam/candidates/

source /home/${STUDENT_ID}/miniconda3/bin/activate jam

echo "Machine:"
hostname
echo "Number of CPUS: "${SLURM_CPUS_PER_TASK}
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
echo "Timestamp: "${TIMESTAMP}
git log -1
git branch
conda info | grep active

# Extract the line with the command to execute
experiment_text_file=$1
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"

benchmark=$(echo $COMMAND | grep -o 'benchmark=[^;]*' | cut -f1 -d ' ' | cut -f2 -d '=')
prob_id=$(echo $COMMAND | grep -o 'problem_id=[^;]*' | cut -f1 -d ' ' | cut -f2 -d '=')
task_id=$(echo $COMMAND | grep -o 'task_id=[^;]*' | cut -f1 -d ' ' | cut -f2 -d '=')

if [ $benchmark = "synth" ]
then
    prob_file=synth_${prob_id}${task_id}.p
    candidates_numpyro_file=numpyro_synthetic_${prob_id}.p
    initial_points_file=synth-starting-points-${prob_id}.p
    related_file=synth_${prob_id}_tune_subset.p
    prior_file=synth_${prob_id}_gamma_prior.json
    shared_file=synth_${prob_id}_supermodel.p
fi
if [ $benchmark = "satellite" ]
then
    prob_file=satellite_${prob_id}_${task_id}.p
    candidates_numpyro_file=numpyro_satellite_${prob_id}.p
    initial_points_file=satellite-starting-points-${prob_id}.p
    related_file=satellite_${prob_id}_tune_subset.p
    prior_file=satellite_${prob_id}_gamma_prior.json
    shared_file=satellite_${prob_id}_supermodel.p
fi

echo "Prob file: "${DATADIR}${prob_file}
echo "Transferring from ${DATADIR}${prob_file} to ${TMPDIR}${prob_file}"
rsync -avuzh ${DATADIR}${prob_file} ${TMPDIR}${prob_file}


echo "Candidate file: "${SAMPLESDIR}${candidates_numpyro_file}
echo "Transferring from ${SAMPLESDIR}${candidates_numpyro_file} to ${TMPDIR}${candidates_numpyro_file}"
rsync -avuzh ${SAMPLESDIR}${candidates_numpyro_file} ${TMPDIR}${candidates_numpyro_file}


echo "Initial points file: "${SAMPLESDIR}${initial_points_file}
echo "Transferring from ${SAMPLESDIR}${initial_points_file} to ${TMPDIR}${initial_points_file}"
rsync -avuzh ${SAMPLESDIR}${initial_points_file} ${TMPDIR}${initial_points_file}


echo "Related tasks file: "${SAMPLESDIR}${related_file}
echo "Transferring from ${SAMPLESDIR}${related_file} to ${TMPDIR}${related_file}"
rsync -avuzh ${SAMPLESDIR}${related_file} ${TMPDIR}${related_file}


echo "Prior gamma file: "${SAMPLESDIR}${prior_file}
echo "Transferring from ${SAMPLESDIR}${prior_file} to ${TMPDIR}${prior_file}"
rsync -avuzh ${SAMPLESDIR}${prior_file} ${TMPDIR}${prior_file}


echo "Shared hps file: "${SAMPLESDIR}${shared_file}
echo "Transferring from ${SAMPLESDIR}${shared_file} to ${TMPDIR}${shared_file}"
rsync -avuzh ${SAMPLESDIR}${shared_file} ${TMPDIR}${shared_file}



COMMAND_FULL="${COMMAND} --res_folder=${JOBTASK} --user_id=${STUDENT_ID}"

echo "Running provided command: ${COMMAND_FULL}"
eval "${COMMAND_FULL}"
echo "Command ran successfully!"

cd ${RESDIR}
if [ $? -eq 0 ]; then
    zip -rDo ${TMPDIR}${JOBTASK}.zip *

    OUT_FOLDER=/home/${STUDENT_ID}/jam/results/${SLURM_ARRAY_JOB_ID}
    echo "Sending results to: ${OUT_FOLDER}/${JOBTASK}"
    mkdir -p ${OUT_FOLDER}

    rsync -avuzh ${TMPDIR}${JOBTASK}.zip ${OUT_FOLDER}/${JOBTASK}.zip
fi
