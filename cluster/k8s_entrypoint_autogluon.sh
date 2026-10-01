#!/bin/bash
# AutoGluon whole-predictor baseline entrypoint -- parallel to k8s_entrypoint.sh, for
# scripts/run_autogluon_baseline.py instead of scripts/run_experiment.py. See that
# script's own module docstring for why this needs a separate jobspec format/entrypoint
# entirely: there is no MODEL/CONFIG_INDEX/NUM_BAG_FOLDS here, only a dataset/target/
# repeat/fold and a preset time budget.
#
# Required env vars (set via build_k8s_job_manifest_autogluon's rendered Job manifest):
#   RESULTS_DIR, CACHE_DIR (already absolute paths under the PVC mount), MIRROR_REPO,
#   USE_GPU, JOBSPEC (path to the mounted ConfigMap volume's jobspec file),
#   TASKS_PER_POD
# DATASET/TARGET_IDX/REPEAT/FOLD/TIME_LIMIT/BUDGET_LABEL/N_REPEATS/N_SPLITS all come
# from this pod's own JOBSPEC lines (cluster/submit_job.py's write_jobspec_autogluon).
#
# TASKS_PER_POD here is expected to be small (unlike the per-model path's default of
# 5000/"one pod per model") -- AutoGluon-extreme's per-task time budgets run up to 4h,
# so batching thousands of tasks sequentially into one pod would make a real sweep take
# impractically long; the submission script is expected to pass a small tasks_per_pod so
# this collapses into many shorter-lived, genuinely parallel pods instead (k8s's own
# Indexed Job `parallelism` setting, see build_k8s_job_manifest_autogluon's own
# docstring).

set -u

TASKS_PER_POD="${TASKS_PER_POD:-1}"

echo "Pod:                  ${HOSTNAME:-unknown}"
echo "Job Completion Index: ${JOB_COMPLETION_INDEX:-0}"
echo "Tasks per pod:        ${TASKS_PER_POD}"
echo "Job started at $(date)"

trap 'echo "Received SIGTERM -- will not start any further tasks in this batch"; exit 1' TERM

[ -f ~/.hf_credentials ] && source ~/.hf_credentials
which python

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

GPU_FLAG=""
[ "${USE_GPU:-0}" = "1" ] && GPU_FLAG="--use-gpu"

FIRST_LINE=$(( ${JOB_COMPLETION_INDEX:-0} * TASKS_PER_POD + 1 ))
LAST_LINE=$(( FIRST_LINE + TASKS_PER_POD - 1 ))
echo "This pod's jobspec lines: ${FIRST_LINE}-${LAST_LINE} of ${JOBSPEC}"

OVERALL_RC=0
TASK_NUM=0
while IFS= read -r LINE; do
    TASK_NUM=$(( TASK_NUM + 1 ))
    [ -z "${LINE}" ] && continue
    read -r DATASET TARGET_IDX REPEAT FOLD TIME_LIMIT BUDGET_LABEL N_REPEATS N_SPLITS <<< "${LINE}"
    echo ""
    echo "=== Task ${TASK_NUM}/${TASKS_PER_POD} (jobspec line $(( FIRST_LINE + TASK_NUM - 1 ))): dataset=${DATASET} target_idx=${TARGET_IDX} repeat=${REPEAT} fold=${FOLD} time_limit=${TIME_LIMIT} budget=${BUDGET_LABEL} ==="

    SCRATCH_ROOT="${CACHE_DIR:-.cache_v1}/scratch_v1"
    TASK_KEY="${HOSTNAME:-pod}_${JOB_COMPLETION_INDEX:-0}_${TASK_NUM}"
    SCRATCH_DIR="${SCRATCH_ROOT}/${TASK_KEY}"
    OPENML_CACHE_DIR="${SCRATCH_ROOT}/openml_${TASK_KEY}"
    mkdir -p "${SCRATCH_DIR}" "${OPENML_CACHE_DIR}"
    export OPENML_CACHE_DIR

    cat > "${SCRATCH_DIR}/job.json" <<JOBMETA
{"pod": "${HOSTNAME:-unknown}", "completion_index": "${JOB_COMPLETION_INDEX:-0}", "task_num": "${TASK_NUM}", "model": "AUTOGLUON-EXTREME-${BUDGET_LABEL}", "dataset": "${DATASET}", "target_idx": "${TARGET_IDX}", "started": "$(date -Iseconds)"}
JOBMETA

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
    TASK_RC=$?
    rm -rf "${SCRATCH_DIR}" "${OPENML_CACHE_DIR}"

    echo "Done: AUTOGLUON-EXTREME-${BUDGET_LABEL} ${DATASET}[${TARGET_IDX}] repeat=${REPEAT} fold=${FOLD} at $(date) (exit=${TASK_RC})"
    if [ "${TASK_RC}" -ne 0 ]; then
        echo "WARNING: task ${TASK_NUM}/${TASKS_PER_POD} failed (exit=${TASK_RC}) -- continuing with the rest of this pod's batch."
        OVERALL_RC=1
    fi
done < <(sed -n "${FIRST_LINE},${LAST_LINE}p" "${JOBSPEC}")

if [ "${TASK_NUM}" -eq 0 ]; then
    echo "No jobspec lines in range ${FIRST_LINE}-${LAST_LINE} -- nothing to do (expected for a partially-filled last pod only if this happens for pod 0)."
fi

echo ""
echo "Pod finished at $(date) (${TASK_NUM} task(s) attempted, overall_exit=${OVERALL_RC})"
exit "${OVERALL_RC}"
