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
COPY pyproject.toml README.md requirements-tabarena-git.txt requirements-models-git.txt ./
COPY src ./src
# --pre: required as of the 2026-09-18 autogluon floor bump (>=1.6.3b20260917,
# a prerelease -- see pyproject.toml's autogluon extra comment for why). Without
# --pre, pip's resolver only considers stable autogluon-core[ray] candidates for
# a transitive requirement line that doesn't itself mention a prerelease,
# conflicting with our own line that does -- confirmed real ResolutionImpossible
# without this flag.
RUN pip install --no-cache-dir --pre -e ".[models,benchmark,tracking]" \
    && pip install --no-cache-dir --pre -r requirements-tabarena-git.txt \
    && pip install --no-cache-dir --pre -r requirements-models-git.txt \
    && pip uninstall -y torchaudio
# torchaudio: the base image's bundled 2.5.1+cu124 build is left over from
# before torch got bumped to 2.14.0+cu130 (same drift class as the
# torchvision fix above), and unlike torchvision, PyPI has published no
# torchaudio release compatible with torch>=2.12 at all (latest is 2.11.0) --
# there is no version to pin to. Confirmed as a real production failure:
# transformers' `is_torchaudio_available()` only checks that the *package*
# is present, not that it actually imports, so MITRA/RAMANFORMER's own
# `transformers` import chain (-> modeling_layers -> processing_utils ->
# audio_utils) unconditionally does `import torchaudio` and hard-crashes with
# "OSError: Could not load this library: .../libtorchaudio.so" (an ABI
# mismatch against the newer torch) on every run. RamanBench has no audio use
# case, so uninstalling it entirely makes the availability check correctly
# report "not available" and transformers skips that import cleanly.
# tabarena moved out of pyproject.toml's [benchmark]/[models] extras (PyPI
# forbids direct-URL dependencies in an uploaded package -- confirmed by the
# v2.0.0 PyPI publish failing with exactly that error) into
# requirements-tabarena-git.txt. Without this second install step, the image
# would have no tabarena at all: every Prep_* model and run_experiment.py
# itself depend on it directly.
#
# requirements-models-git.txt (TabFM, SAP-RPT-OSS, OrionMSP/tabtune) was
# ALSO never wired in here -- confirmed as a real production failure: every
# single TABFM task on the k8s cluster failed with "ModuleNotFoundError: No
# module named 'tabfm'" (100% failure rate, not an edge case). Without this
# install step, Prep_TABFM/Prep_SAP_RPT_OSS/Prep_ORIONMSP are silently
# unavailable in every image built from this Dockerfile.

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
