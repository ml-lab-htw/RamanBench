#!/bin/bash
#SBATCH --output=.logs/%A_%a_%x.log
#SBATCH --error=.logs/%A_%a_%x.log
#SBATCH --signal=B:USR1@120
#SBATCH --job-name=RamanBenchV1
#
# Generic per-(model,dataset,target,seed,config-index) job. Resource flags
# (--mem, --gres/--gpus-per-task, --account, --partition, --mail-user, ...)
# are supplied on the sbatch CLI by cluster/submit_job.py, not baked in here
# -- this file has no institution-specific values.
#
# Required env vars (set via --export by submit_job.py):
#   MODEL, N_SPLITS, NUM_RANDOM_CONFIGS, NUM_BAG_FOLDS, TIME_LIMIT, RESULTS_DIR,
#   CACHE_DIR, MIRROR_REPO, USE_GPU, JOBSPEC, ACTIVATION, CONDA_ENV, VENV_PATH, WORKSPACE
# DATASET/TARGET_IDX/REPEAT/FOLD/CONFIG_INDEX/N_REPEATS come from this task's own
# JOBSPEC line instead -- one array can span multiple (dataset, target) pairs for
# the same MODEL (a single sbatch call's resource flags are per-model, not per-task,
# so MODEL itself must stay array-wide), not just multiple (repeat, fold, config)
# tuples for one fixed (dataset, target).

echo "SLURM Job ID:        $SLURM_JOB_ID"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "SLURM Nodes:         $SLURM_NODELIST"
echo "Model:               $MODEL"
echo "Job started on $(hostname) at $(date)"

# --- Deterministic scratch dir for this task's AutoGluon predictor artifacts.
#     Named by job id (not a random uuid) so a janitor sweep (cluster/janitor.py)
#     can cross-reference it against squeue/sacct and safely identify orphans
#     left behind by jobs killed before this script's own cleanup trap (or
#     AutoGluon's own post-fit cleanup, which handles the normal-exit case)
#     could run. This is the fix for jobs that vanish via timeout/OOM/scancel
#     rather than finishing normally -- the root cause of the 27TB problem. ---
SCRATCH_ROOT="${CACHE_DIR:-.cache_v1}/scratch_v1"
SCRATCH_DIR="${SCRATCH_ROOT}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${SCRATCH_DIR}"

# --- Per-job OpenML local-dataset cache, same reasoning as SCRATCH_DIR above.
#     TabArena's UserTask.create_local_openml_task keys its on-disk cache
#     (~/.cache/openml/local/datasets/<hash>/data.pkl.py3) by dataset+target
#     only, not by repeat/fold, so concurrent jobs on the same task
#     technically share this file. A real batch's crashes turned out to
#     have a different root cause (missing/NaN labels, oversized
#     num_bag_folds on tiny datasets -- both fixed in run_experiment.py),
#     not a confirmed corruption here, but TabArena's own
#     tabflow_slurm/run_tabarena_experiment.py treats this cache dir as an
#     explicit, deliberately job-scoped parameter too -- keeping it isolated
#     is cheap, well-precedented hygiene, not just theoretical caution. ---
export OPENML_CACHE_DIR="${SCRATCH_ROOT}/openml_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${OPENML_CACHE_DIR}"

cleanup() {
    echo "Cleanup: removing ${SCRATCH_DIR} and ${OPENML_CACHE_DIR}"
    rm -rf "${SCRATCH_DIR}" "${OPENML_CACHE_DIR}"
}
# Fires on ANY exit path (normal completion, error, or a signal-induced exit
# below) -- this is the guarantee today's Python-only cleanup lacks.
trap cleanup EXIT
# SLURM sends SIGUSR1 120s before the time limit (see --signal above), giving
# a chance to exit (and thus trigger the EXIT trap above) before SIGKILL.
trap 'echo "Received pre-timeout signal (120s to go) -- exiting for cleanup"; exit 1' USR1

# --- Environment activation ---
if [ "${ACTIVATION:-conda}" = "conda" ]; then
    source ~/.bashrc
    conda activate "${CONDA_ENV:-raman_bench}"
else
    source "${WORKSPACE:-.}/${VENV_PATH:-.venv}/bin/activate"
fi
[ -f ~/.hf_credentials ] && source ~/.hf_credentials
which python

# Enabled only now, AFTER sourcing ~/.bashrc/conda-activate/.hf_credentials --
# those (and /etc/bashrc, transitively) reference variables that are
# legitimately unset outside an interactive login shell, and `set -u` before
# this point makes any of them fatal (confirmed in practice: /etc/bashrc's own
# BASHRCSOURCED guard tripped this exact error on a real deployment).
set -u

# --- cd into the workspace. Deliberately NOT pulling anything here anymore:
#     jobs in the same array batch must all run against the exact same,
#     already-resolved code/dependency versions -- pulling per-job meant
#     concurrent tasks could race each other and silently run different code,
#     and fought against "rely on a released version" (a checkout that's
#     live-pulled mid-batch isn't pinned to anything). Refreshing
#     raman-data/raman-bench and this checkout is now cluster-agent's job,
#     done once before submitting the batch -- see cluster/refresh_deps.py. ---
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

# --- Read this array task's (dataset, target, seed, config_index) from the
#     jobspec line -- dataset/target_idx/n_repeats are per-line rather than
#     --export env vars so one array can cover many (dataset, target) pairs
#     for this MODEL, not just many (repeat, fold, config) tuples for one. ---
LINE_NO=$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))
read -r DATASET TARGET_IDX REPEAT FOLD CONFIG_INDEX N_REPEATS <<< "$(sed -n "${LINE_NO}p" "${JOBSPEC}")"
echo "Task line ${LINE_NO} of ${JOBSPEC}: dataset=${DATASET} target_idx=${TARGET_IDX} repeat=${REPEAT} fold=${FOLD} config_index=${CONFIG_INDEX}"

cat > "${SCRATCH_DIR}/job.json" <<JOBMETA
{"job_id": "${SLURM_JOB_ID}", "array_task_id": "${SLURM_ARRAY_TASK_ID:-0}", "model": "${MODEL}", "dataset": "${DATASET}", "target_idx": "${TARGET_IDX}", "started": "$(date -Iseconds)"}
JOBMETA

GPU_FLAG=""
[ "${USE_GPU:-0}" = "1" ] && GPU_FLAG="--use-gpu"

python scripts/run_experiment.py \
    --dataset "${DATASET}" \
    --target-idx "${TARGET_IDX}" \
    --model "${MODEL}" \
    --repeat "${REPEAT}" \
    --fold "${FOLD}" \
    --n-repeats "${N_REPEATS}" \
    --n-splits "${N_SPLITS}" \
    --config-index "${CONFIG_INDEX}" \
    --num-random-configs "${NUM_RANDOM_CONFIGS}" \
    --num-bag-folds "${NUM_BAG_FOLDS}" \
    --time-limit "${TIME_LIMIT}" \
    --results-dir "${RESULTS_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --mirror-repo "${MIRROR_REPO}" \
    --scratch-dir "${SCRATCH_DIR}" \
    ${GPU_FLAG}
RC=$?

# Propagate the real exit code -- without this, a crashed run_experiment.py
# still reports SLURM job success (no `set -e`, and this echo always
# succeeds). Confirmed in practice: sacct showed COMPLETED for a real batch
# of jobs that had actually crashed on missing (NaN) labels or an
# over-large num_bag_folds for tiny datasets, both silently, until this was
# added and results.pkl presence was checked directly instead of trusting
# sacct's state.
echo "Done: ${MODEL} ${DATASET}[${TARGET_IDX}] repeat=${REPEAT} fold=${FOLD} config_index=${CONFIG_INDEX} at $(date) (exit=${RC})"
exit $RC
