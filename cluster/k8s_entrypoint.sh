#!/bin/bash
# Generic per-(model,dataset,target,seed,config-index) job entrypoint for a
# Kubernetes Indexed Job -- the k8s equivalent of run_experiment.sbatch.
# Baked into the container image's CMD/ENTRYPOINT or passed as the pod's
# `command`; resource requests/limits, image, and volume mounts are supplied
# by submit_job.py's rendered Job manifest, not baked in here -- this file
# has no institution-specific values.
#
# Required env vars (set via the Job manifest's container `env`):
#   MODEL, N_SPLITS, NUM_RANDOM_CONFIGS, NUM_BAG_FOLDS, TIME_LIMIT, RESULTS_DIR,
#   CACHE_DIR (both already absolute paths under the PVC mount -- see
#   submit_job.py's submit_jobs_k8s), MIRROR_REPO, USE_GPU, JOBSPEC (path to
#   the mounted ConfigMap volume's jobspec file), TASKS_PER_POD
# DATASET/TARGET_IDX/REPEAT/FOLD/CONFIG_INDEX/N_REPEATS come from this pod's
# own JOBSPEC lines, indexed by $JOB_COMPLETION_INDEX (k8s's analogue of
# $SLURM_ARRAY_TASK_ID, automatically set by an Indexed Job -- see
# https://kubernetes.io/docs/tasks/job/indexed-parallel-processing-static/).
#
# TASKS_PER_POD > 1 batches multiple jobspec lines into one pod, run
# SEQUENTIALLY in a loop below -- courtesy default for a shared, multi-tenant
# k8s cluster, where launching one pod per individual (repeat, fold,
# config) task -- as fine-grained as the SLURM array path -- means thousands
# of short-lived pods showing up in `kubectl get pods` for every other user
# on the cluster, even though the total GPU-*time* used is identical either
# way. Batching trades that pod-count courtesy for coarser fault isolation:
# unlike the one-task-per-pod case (where a crash just fails that one array
# slot), a crash in the middle of a batch must NOT take down the remaining
# tasks in the same pod -- so each task's failure is caught and logged, and
# the loop continues; the pod's own exit code is non-zero if ANY task in its
# batch failed, so `kubectl get jobs`/an unfinished completions count still
# surfaces that something needs attention.
#
# Unlike run_experiment.sbatch, this script does NOT activate a conda/venv
# environment -- the container image is expected to already have raman-bench
# (and raman-data) installed. It also doesn't `git pull`: the image is built
# from a pinned checkout, so every task in a submission runs identical code,
# same reasoning as the SLURM path's "no per-job pull" comment.

set -u

TASKS_PER_POD="${TASKS_PER_POD:-1}"

echo "Pod:                  ${HOSTNAME:-unknown}"
echo "Job Completion Index: ${JOB_COMPLETION_INDEX:-0}"
echo "Tasks per pod:        ${TASKS_PER_POD}"
echo "Model:                ${MODEL}"
echo "Job started at $(date)"

# Kubernetes sends SIGTERM (not SIGUSR1 like SLURM) before SIGKILL, with the
# pod's terminationGracePeriodSeconds as the delay. This doesn't stop
# mid-task (run_experiment.py isn't interrupted early -- same as the SLURM
# path never interrupting a fold mid-fit), just logs so a truncated batch is
# visible in the pod's own logs rather than silently vanishing.
trap 'echo "Received SIGTERM -- will not start any further tasks in this batch"; exit 1' TERM

[ -f ~/.hf_credentials ] && source ~/.hf_credentials
which python

# Unlike run_experiment.sbatch's shared-filesystem checkout, this image bakes
# code into /app (the Dockerfile's WORKDIR) -- there is no workspace checkout
# to `cd` into here, and RESULTS_DIR/CACHE_DIR now arrive already resolved to
# absolute paths under the PVC mount (see submit_job.py's submit_jobs_k8s),
# so cwd staying at /app is exactly right: that's where scripts/run_experiment.py
# actually lives. An earlier version of this script did `cd "$WORKSPACE"`
# (a PVC path with no code at all) before invoking a *relative*
# `python scripts/run_experiment.py` -- every task "succeeded" while silently
# writing results.pkl to the pod's own ephemeral container filesystem instead
# of the PVC, invisible until the pod was gone and the results were gone with
# it. Confirmed happening in practice on this project's first real smoke test.
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

# --- This pod's slice of the jobspec: TASKS_PER_POD consecutive lines
#     starting at JOB_COMPLETION_INDEX * TASKS_PER_POD (0-based), i.e. pod 0
#     takes lines 1..K, pod 1 takes lines K+1..2K, etc. -- so `completions`
#     in the Job manifest is ceil(total_tasks / TASKS_PER_POD), not
#     total_tasks (see submit_job.submit_jobs_k8s). The last pod's slice may
#     run past EOF; `sed` just returns fewer lines, handled by the loop below. ---
FIRST_LINE=$(( ${JOB_COMPLETION_INDEX:-0} * TASKS_PER_POD + 1 ))
LAST_LINE=$(( FIRST_LINE + TASKS_PER_POD - 1 ))
echo "This pod's jobspec lines: ${FIRST_LINE}-${LAST_LINE} of ${JOBSPEC}"

OVERALL_RC=0
TASK_NUM=0
while IFS= read -r LINE; do
    TASK_NUM=$(( TASK_NUM + 1 ))
    [ -z "${LINE}" ] && continue
    read -r DATASET TARGET_IDX REPEAT FOLD CONFIG_INDEX N_REPEATS TASK_TIME_LIMIT TASK_MAX_TRAIN_SAMPLES <<< "${LINE}"
    EFFECTIVE_TIME_LIMIT="${TASK_TIME_LIMIT:-${TIME_LIMIT}}"
    # TASK_MAX_TRAIN_SAMPLES (8th field) is a per-dataset row-subsampling
    # override (cluster/scope_default.json's max_train_samples_overrides,
    # e.g. mlrod) -- unset/empty for every other dataset, in which case
    # --max-train-samples is simply omitted (run_experiment.py's own default,
    # no subsampling).
    MAX_TRAIN_SAMPLES_FLAG=()
    if [ -n "${TASK_MAX_TRAIN_SAMPLES}" ]; then
        MAX_TRAIN_SAMPLES_FLAG=(--max-train-samples "${TASK_MAX_TRAIN_SAMPLES}")
    fi
    echo ""
    echo "=== Task ${TASK_NUM}/${TASKS_PER_POD} (jobspec line $(( FIRST_LINE + TASK_NUM - 1 ))): dataset=${DATASET} target_idx=${TARGET_IDX} repeat=${REPEAT} fold=${FOLD} config_index=${CONFIG_INDEX} time_limit=${EFFECTIVE_TIME_LIMIT} ==="

    # Deterministic per-task scratch dir, same orphan-cleanup rationale as
    # run_experiment.sbatch -- keyed by pod hostname AND this task's own
    # position within the batch (not just the pod), since a batched pod runs
    # multiple tasks in sequence, each needing its own AutoGluon scratch space.
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
exit $OVERALL_RC
