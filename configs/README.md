# configs

One folder per benchmark version. Each is self-contained: nothing in `v2/` reads a file of `v1/`.

| Folder | What it is |
|---|---|
| `v0.1/` | The first benchmark's config, kept as a record. `benchmark.json` is read by `Leaderboard` when it scores the v0.1 data; `datasets/` and `models/` are the lists it points to (the model lists are also what the model-registry tests check). The pipeline that ran it is gone. |
| `v1/` | Benchmark v1 (`main`): `scope_default.json` (what a full sweep runs), `target_list.json` (every target), `exclusions.json` (+ `EXCLUDED_TARGETS.md`, the human-readable view), `scope_core.json` and `core_target_list.json` (the core subset), `datasets/` (the dataset lists), `models.json`. |
| `v2/` | Benchmark v2 (`v2` branch): the same shared files plus `protocol.json` (every task and the settings every model runs under); `datasets/all.json` lists every v2 dataset and `datasets/new.json` the ones v1 did not have. See `v2/README.md`. |

The same names mean the same thing in every version: `scope_default.json`, `target_list.json`, `exclusions.json`, `datasets/`.
