# DepthVLA-OFT Final Robotics Project

This repository contains the final course project code and report for
**DepthVLA-OFT: Diagnosing and Rebuilding RGB-D Action Grounding for VLA
Manipulation**.

## Final Claim

The project does **not** claim that end-to-end OpenVLA-OFT RGB-D outperforms
RGB-only OpenVLA-OFT. Instead, it succeeds as a diagnostic robotics project:

- LIBERO/OpenVLA-OFT shows a saturated RGB-only benchmark and explains why
  lightweight optional depth fusion can be ignored.
- RLBench/OpenVLA-OFT shows that depth can enter a geometric bottleneck, but the
  tested lightweight adapters did not produce rollout task success.
- ManiSkill3 PickCube provides the main positive result: when raw point-cloud
  geometry enters the primary learned action decoder, normal point clouds reach
  `20/60` closed-loop successes, while null and cross-demo controls reach
  `1/60` each. Matched no-depth baselines reach `1/60` and `3/60`.

## Main Materials

- `main1.pdf`: final report PDF.
- `main1.tex`: final report source.
- `IEEEtran.cls`: LaTeX class file used by the RSS-style template.
- `LIBERO/`: LIBERO benchmark code used for RGB-D regeneration and clean
  saturation checks.
- `openvla-oft/`: OpenVLA-OFT-based RGB-D training, evaluation, diagnostics,
  RLBench experiments, and ManiSkill3 point-cloud policies.

## Important Code Paths

- `openvla-oft/vla-scripts/finetune_depthvla.py`: DepthVLA/OpenVLA-OFT training
  and fusion variants.
- `openvla-oft/vla-scripts/depth_signal_probe.py`: depth-action coupling
  diagnostics.
- `openvla-oft/experiments/robot/libero/`: LIBERO RGB-D regeneration and
  evaluation helpers.
- `openvla-oft/experiments/robot/rlbench/`: RLBench conversion, rollout, and
  geometry probes.
- `openvla-oft/experiments/robot/maniskill/`: ManiSkill3 PickCube point-cloud
  data collection, training, and evaluation scripts.
- `openvla-oft/scripts/collect_depthvla_final_results.py`: final result
  collection helper.

Large generated datasets, checkpoints, virtual environments, cache directories,
and unrelated project folders are excluded from git.
