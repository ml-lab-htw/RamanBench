# Image for running RamanBench experiments as Kubernetes Job pods (see
# cluster/profiles/k8s_example.yaml, cluster/k8s_entrypoint.sh,
# cluster/submit_job.py's k8s backend). Not used by the SLURM path, which
# instead activates a conda/venv environment on the cluster's own shared
# filesystem -- see cluster/profiles/{htw,tu}.yaml.
#
# Build and push (fill in your own registry):
#   docker build -t <registry>/<you>/raman-bench:latest .
#   docker push <registry>/<you>/raman-bench:latest
# Then set `image: "<registry>/<you>/raman-bench:latest"` in your k8s profile.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
# build-essential (gcc/g++): the base image is deliberately a "runtime" (not
# "devel") variant with no C compiler -- fine until torch>=2.13 (forced by
# causilo's floor, see pyproject.toml) pulled in a triton version whose
# kernels are JIT-compiled at call time via `triton.runtime.build`, which
# shells out to `cc`. Confirmed as a real production failure: every REALMLP
# task failed with "RuntimeError: Failed to find C compiler" until this was
# added -- the earlier torchvision version-mismatch bug (also from the same
# torch bump) was masking this one, since tasks never got far enough to reach
# triton's JIT path before this fix.

WORKDIR /app

# Install the released package plus its optional extras used for full
# benchmark runs -- mirrors the SLURM clusters' `uv pip install -e ".[models]"`
# / conda env setup (see raman_bench_paper/cluster/profiles/htw.yaml).
COPY pyproject.toml README.md ./
COPY src ./src
# --pre: required as of the 2026-09-18 autogluon floor bump (>=1.6.3b20260917,
# a prerelease -- see pyproject.toml's autogluon extra comment for why). Without
# --pre, pip's resolver only considers stable autogluon-core[ray] candidates for
# a transitive requirement line that doesn't itself mention a prerelease,
# conflicting with our own line that does -- confirmed real ResolutionImpossible
# without this flag.
RUN pip install --no-cache-dir --pre -e ".[models,benchmark,tracking]"

# scripts/ and cluster/ are what run_experiment.py and k8s_entrypoint.sh need
# at runtime; the rest of the checkout (docs, tests, configs) isn't needed in
# the image -- the real, git-tracked source of truth stays on the PVC
# (see cluster/profiles/k8s_example.yaml's `workspace`) for reproducibility
# bookkeeping, but the image only needs enough to run one task.
COPY scripts ./scripts
COPY cluster ./cluster
COPY configs ./configs

RUN chmod +x cluster/k8s_entrypoint.sh

ENTRYPOINT ["/bin/bash", "cluster/k8s_entrypoint.sh"]
