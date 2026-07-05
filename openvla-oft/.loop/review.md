# Review Notes

## Current Verification Standard

Do not claim RGB-D beats RGB until all conditions hold:

1. Matched RGB-only and RGB-D checkpoints are trained on the same dataset.
2. RGB-D normal exceeds RGB-only in closed-loop rollout.
3. RGB-D normal exceeds RGB-D null and shuffle depth.
4. Improvement is not only a single cherry-picked episode.
5. Commands and result JSON files are recorded.

## Current Risk Register

- RLBench policy may fail before any depth comparison becomes meaningful.
- Depth branch currently passes offline keypose probe but not action-level causal diagnostic.
- `/root/autodl-tmp` has limited free disk; run outputs should stay under `/root/runs_*`.
- Parallel 7B diagnostics can OOM; run them serially.
