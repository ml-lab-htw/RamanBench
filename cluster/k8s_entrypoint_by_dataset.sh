#!/bin/bash
# Dataset-centric counterpart to k8s_entrypoint.sh: iterates many MODELS
# against one (or a few) fixed dataset(s), instead of one fixed MODEL against
# many datasets. Built for exactly the case the per-model path handles
# badly: adding/recomputing one or a few datasets across the whole model
# roster used to mean one pod PER MODEL (tens of pods for one dataset) --
# this needs just one pod per (Python-version image, CPU/GPU tier) group
# instead, since that's the only axis that actually forces a separate pod
# (different container image, or GPU vs CPU-only resource shape).
#
# Not a generalization of k8s_entrypoint.sh's jobspec format -- a deliberately
# separate, parallel format/script, so the existing (well-tested, currently
# running) per-model path is never at risk of a regression from this one.
#
# Required env vars (set via the Job manifest's container `env`):
#   N_SPLITS, NUM_RANDOM_CONFIGS, NUM_BAG_FOLDS, TIME_LIMIT, RESULTS_DIR,
#   CACHE_DIR, MIRROR_REPO, USE_GPU, JOBSPEC, TASKS_PER_POD
# MODEL/DATASET/TARGET_IDX/REPEAT/FOLD/CONFIG_INDEX/N_REPEATS all come from
# this pod's own JOBSPEC lines (MODEL is the new 1st field here -- every
# other field is in the same order/meaning as k8s_entrypoint.sh's format,
# just shifted one column right). See submit_dataset_sweep.py for how the
# jobspec is built and how models are grouped into pods.

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
    read -r MODEL DATASET TARGET_IDX REPEAT FOLD CONFIG_INDEX N_REPEATS TASK_TIME_LIMIT TASK_MAX_TRAIN_SAMPLES <<< "${LINE}"
    EFFECTIVE_TIME_LIMIT="${TASK_TIME_LIMIT:-${TIME_LIMIT}}"
    MAX_TRAIN_SAMPLES_FLAG=()
    if [ -n "${TASK_MAX_TRAIN_SAMPLES}" ]; then
        MAX_TRAIN_SAMPLES_FLAG=(--max-train-samples "${TASK_MAX_TRAIN_SAMPLES}")
    fi
    echo ""
    echo "=== Task ${TASK_NUM}/${TASKS_PER_POD} (jobspec line $(( FIRST_LINE + TASK_NUM - 1 ))): model=${MODEL} dataset=${DATASET} target_idx=${TARGET_IDX} repeat=${REPEAT} fold=${FOLD} config_index=${CONFIG_INDEX} time_limit=${EFFECTIVE_TIME_LIMIT} ==="

    SCRATCH_ROOT="${CACHE_DIR:-.cache_v1}/scratch_v1"
    TASK_KEY="${HOSTNAME:-pod}_${JOB_COMPLETION_INDEX:-0}_${TASK_NUM}"
    SCRATCH_DIR="${SCRATCH_ROOT}/${TASK_KEY}"
    OPENML_CACHE_DIR="${SCRATCH_ROOT}/openml_${TASK_KEY}"
    mkdir -p "${SCRATCH_DIR}" "${OPENML_CACHE_DIR}"
    export OPENML_CACHE_DIR

    cat > "${SCRATCH_DIR}/job.json" <<JOBMETA
{"pod": "${HOSTNAME:-unknown}", "completion_index": "${JOB_COMPLETION_INDEX:-0}", "task_num": "${TASK_NUM}", "model": "${MODEL}", "dataset": "${DATASET}", "target_idx": "${TARGET_IDX}", "started": "$(date -Iseconds)"}
JOBMETA

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
        --time-limit "${EFFECTIVE_TIME_LIMIT}" \
        --results-dir "${RESULTS_DIR}" \
        --cache-dir "${CACHE_DIR}" \
        --mirror-repo "${MIRROR_REPO}" \
        --scratch-dir "${SCRATCH_DIR}" \
        "${MAX_TRAIN_SAMPLES_FLAG[@]}" \
        ${GPU_FLAG}
    TASK_RC=$?
    rm -rf "${SCRATCH_DIR}" "${OPENML_CACHE_DIR}"

    echo "Done: ${MODEL} ${DATASET}[${TARGET_IDX}] repeat=${REPEAT} fold=${FOLD} config_index=${CONFIG_INDEX} at $(date) (exit=${TASK_RC})"
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
