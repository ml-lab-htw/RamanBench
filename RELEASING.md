# Releasing raman-bench

## Version numbers

`MAJOR.MINOR.PATCH`, where **MAJOR is the benchmark version**: every `1.x.y` release runs
and scores RamanBench v1, and benchmark v2 will be `2.x.y`. This rule holds since 1.2.0.

| Bump | When | Examples |
|---|---|---|
| **MAJOR** | A new benchmark version: anything that makes new scores incomparable with the published ones | datasets or targets added/removed from the protocol, different outer folds or repeats, a different metric, time budget or scoring (Elo/imputation) rule |
| **MINOR** | Anything new within the same benchmark, including an API break | a new model, a new feature or CLI command, rebuilt reference results (reruns, new models), removing or renaming a public function |
| **PATCH** | Fixes that change nothing above | a bug fix, a packaging or dependency fix, docs |

Notes:

- An API break is a MINOR bump, not a MAJOR one. Say so under **Removed** or **Changed**
  in CHANGELOG.md so users who pin `>=1.x,<2` see it.
- Leaderboard numbers can change within a major version (new models, reruns of broken
  folds); the protocol behind them can't.
- `2.0.0` and `2.0.1` were released before this rule and are benchmark v1 (2.0.1 is
  yanked on PyPI). Version numbers can never be reused on PyPI, so benchmark v2 starts at
  **2.1.0**.
- Downstream projects pin the benchmark with `raman-bench>=1.2,<2`.

## Release steps

1. On a branch, set `version` in `pyproject.toml` and `__version__` in
   `src/raman_bench/__init__.py`.
2. In CHANGELOG.md, rename `## [Unreleased]` to `## [X.Y.Z] — YYYY-MM-DD` and add a new
   empty `## [Unreleased]` above it.
3. Open a PR titled `Release vX.Y.Z: <summary>` and merge it once CI passes.
4. Tag the merge commit and create the GitHub Release:

   ```bash
   git switch main && git pull
   git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z
   gh release create vX.Y.Z --generate-notes
   ```

   Creating the **Release** runs `.github/workflows/publish.yml`, which publishes to PyPI
   (trusted publisher). Pushing the tag alone publishes nothing.
5. Check the workflow run and https://pypi.org/project/raman-bench/.

PyPI rejects direct (`git+`) dependencies, which is why `tabarena` is installed from
`requirements-tabarena-git.txt` and not declared in `pyproject.toml`. Don't add a git
dependency to the package metadata.

## Yanking

A broken or misnumbered release is yanked, not deleted: pypi.org → raman-bench → Manage →
Releases → the version → Options → Yank, with a reason. pip then skips it unless it is
pinned exactly. Publish the replacement first, so `pip install raman-bench` never falls
back to an older release.
