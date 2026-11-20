# Contributor agents

Six Claude Code agents handle routine maintenance across the three repos, so a contributor
doesn't need any private tooling. Each is a `.claude/agents/*.md` file in the repo it works on.

| Agent | Repo | Responsibility |
|---|---|---|
| `dataset-agent` | `raman_data` | Onboard a new dataset: pick the right loader, add a `DatasetInfo` entry, populate `group_ids`/`has_missing_labels` if applicable, sync to the HF mirror. |
| `model-agent` | `RamanBench` (this repo) | Add a new model via the per-directory convention, test it, ask whether to also propose it upstream to TabArena, then run it across the benchmark (cluster or local). |
| `cluster-agent` | `RamanBench` (public) + `raman_bench_paper` (private profiles) | Fleet management: submit job arrays, detect stalled or cancelled tasks and resubmit, run `cluster/janitor.py`'s disk-cleanup sweep. |
| `leaderboard-agent` | `raman_bench_paper` | Regenerate leaderboard CSVs and figures from completed results; always shows a diff and asks permission before publishing to the live HF Space. |
| `hf-frontend-agent` | `HF_spaces/RamanBench` | Frontend work on the public Gradio leaderboard Space. |
| `docs-agent` | `RamanBench` (this repo) | Keeps README and CONTRIBUTING in sync across all repos as things change. |

A typical handoff runs dataset-agent, model-agent (optional), cluster-agent, leaderboard-agent
(which asks before publishing), hf-frontend-agent. docs-agent runs on its own after structural
changes.

To add a model or a dataset with an agent, open the repo in Claude Code and say:

```
Add my model to RamanBench, test it, and run it across the benchmark.
Add my dataset to RamanBench and make it benchmarkable.
```

The manual route is in [CONTRIBUTING.md](../CONTRIBUTING.md) and [NEW_DATASETS.md](../NEW_DATASETS.md).
The [live leaderboard](https://huggingface.co/spaces/HTW-KI-Werkstatt/RamanBench) has its own
"How to Contribute" section.
