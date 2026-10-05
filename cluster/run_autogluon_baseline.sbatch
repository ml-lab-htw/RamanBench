#!/bin/bash
#SBATCH --output=.logs/%A_%a_%x.log
#SBATCH --error=.logs/%A_%a_%x.log
#SBATCH --signal=B:USR1@120
#SBATCH --job-name=RamanBenchV1Autogluon
#
# AutoGluon-extreme whole-predictor baseline, SLURM counterpart to
# cluster/run_experiment.sbatch (the per-model path) -- drives
# scripts/run_autogluon_baseline.py instead of scripts/run_experiment.py,
# reading the AutoGluonTask jobspec format (8 fields: dataset, target_idx,
# repeat, fold, time_limit, budget_label, n_repeats, n_splits -- see
# submit_job.write_jobspec_autogluon) rather than run_experiment.sbatch's
# 6-8 field per-model format. Resource flags are supplied on the sbatch CLI
# by cluster/submit_autogluon_baseline.py, not baked in here, same convention
# as run_experiment.sbatch.
#
# Required env vars (set via --export): RESULTS_DIR, CACHE_DIR, MIRROR_REPO,
# USE_GPU, JOBSPEC, ACTIVATION, CONDA_ENV, VENV_PATH, WORKSPACE.
# DATASET/TARGET_IDX/REPEAT/FOLD/TIME_LIMIT/BUDGET_LABEL/N_REPEATS/N_SPLITS
# all come from this task's own JOBSPEC line.

echo "SLURM Job ID:        $SLURM_JOB_ID"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "SLURM Nodes:         $SLURM_NODELIST"
echo "Job started on $(hostname) at $(date)"

SCRATCH_ROOT="${CACHE_DIR:-.cache_v1}/scratch_v1"
SCRATCH_DIR="${SCRATCH_ROOT}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${SCRATCH_DIR}"

export OPENML_CACHE_DIR="${SCRATCH_ROOT}/openml_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${OPENML_CACHE_DIR}"

cleanup() {
    echo "Cleanup: removing ${SCRATCH_DIR} and ${OPENML_CACHE_DIR}"
    rm -rf "${SCRATCH_DIR}" "${OPENML_CACHE_DIR}"
}
trap cleanup EXIT
trap 'echo "Received pre-timeout signal (120s to go) -- exiting for cleanup"; exit 1' USR1

if [ "${ACTIVATION:-conda}" = "conda" ]; then
    source ~/.bashrc
    conda activate "${CONDA_ENV:-raman_bench}"
else
    source "${WORKSPACE:-.}/${VENV_PATH:-.venv}/bin/activate"
fi
[ -f ~/.hf_credentials ] && source ~/.hf_credentials
which python

set -u

if [ -n "${WORKSPACE:-}" ]; then
    cd "${WORKSPACE}"
fi
echo "RamanBench checkout: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
python -c "
from importlib.metadata import version, PackageNotFoundError
for pkg in ('raman-data', 'raman-bench'):
    try:
        print(f'{pkg}: {version(pkg)}')
    except PackageNotFoundError:
        print(f'{pkg}: (not installed)')
"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
if [ "${USE_GPU:-0}" = "0" ]; then
    export CUDA_VISIBLE_DEVICES=""
fi

LINE_NO=$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))
read -r DATASET TARGET_IDX REPEAT FOLD TIME_LIMIT BUDGET_LABEL N_REPEATS N_SPLITS <<< "$(sed -n "${LINE_NO}p" "${JOBSPEC}")"
echo "Task line ${LINE_NO} of ${JOBSPEC}: dataset=${DATASET} target_idx=${TARGET_IDX} repeat=${REPEAT} fold=${FOLD} time_limit=${TIME_LIMIT} budget=${BUDGET_LABEL}"

cat > "${SCRATCH_DIR}/job.json" <<JOBMETA
{"job_id": "${SLURM_JOB_ID}", "array_task_id": "${SLURM_ARRAY_TASK_ID:-0}", "model": "AUTOGLUON-EXTREME-${BUDGET_LABEL^^}", "dataset": "${DATASET}", "target_idx": "${TARGET_IDX}", "started": "$(date -Iseconds)"}
JOBMETA

GPU_FLAG=""
[ "${USE_GPU:-0}" = "1" ] && GPU_FLAG="--use-gpu"

python scripts/run_autogluon_baseline.py \
    --dataset "${DATASET}" \
    --target-idx "${TARGET_IDX}" \
    --repeat "${REPEAT}" \
    --fold "${FOLD}" \
    --time-limit "${TIME_LIMIT}" \
    --budget-label "${BUDGET_LABEL}" \
    --n-repeats "${N_REPEATS}" \
    --n-splits "${N_SPLITS}" \
    --results-dir "${RESULTS_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --mirror-repo "${MIRROR_REPO}" \
    --scratch-dir "${SCRATCH_DIR}" \
    ${GPU_FLAG}
RC=$?

echo "Done: AUTOGLUON-EXTREME-${BUDGET_LABEL} ${DATASET}[${TARGET_IDX}] repeat=${REPEAT} fold=${FOLD} at $(date) (exit=${RC})"
exit $RC
