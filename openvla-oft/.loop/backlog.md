# Loop Backlog

## Highest Priority

- [x] T-001: Finish reach-only RGB-only overfit training.
  - Verify: checkpoint exists and `eval-rgb` on `reach_target` with horizon 150 completes.
- [x] T-002: Train reach-only RGB-D dense/keypose model.
  - Verify: checkpoint exists and offline action diagnostic runs for normal/null/shuffle.
- [x] T-003: Run reach-only closed-loop comparison.
  - Verify: JSON results exist for RGB-only, RGB-D normal, RGB-D null, RGB-D shuffle.
- [x] T-004: Repair harmful RGB-D fusion before stable6 scaling.
  - Verify: safe RGB-anchor RGB-D normal reaches `1/1` on `reach_target` and does not underperform null/cross-sample.
- [x] T-005: Add cross-sample depth corruption for dense-point rollout/diagnostic.
  - Verify: `diagnose-rgbd-cross-sample` and `eval-rgbd-cross-sample` smoke complete.
- [x] T-006: Build the next causal depth gate on a 3D-sensitive task/action target, not `reach_target`.
  - Result: completed on RLBench `open_drawer`; depth has offline spatial signal, but all policy gates failed to produce a defensible rollout gain.
- [x] T-007: Train and evaluate `open_drawer` RGB-only baseline.
  - Verify: closed-loop eval result exists for `open_drawer` with horizon `150-200`.
- [x] T-008: Train safe RGB-D `open_drawer` from the RGB-only anchor.
  - Verify: normal/null/cross-sample rollout and offline diagnostics exist.
- [x] T-009: Test keypose-conditioned residual on `open_drawer`.
  - Result: trained to `5000` steps, but failed causal gate.
  - Evidence: paired normal-vs-cross-sample action delta is exactly `0.0`; normal RMSE `0.0137849`, null RMSE `0.0137148`, cross-sample RMSE `0.0137849`.
  - Decision: do not run long rollout or scale this recipe.
- [x] T-010: Replace keypose residual with explicit spatial action grounding.
  - Candidate A: BridgeVLA-style projected multi-view heatmap from depth/point cloud.
  - Candidate B: Act3D/PerAct-style coarse-to-fine 3D action map.
  - Candidate C: PointACT-style multi-scale point-action attention.
  - Result: projected heatmap, point-action, and primary waypoint-action variants were implemented and tested; none passed the final causal rollout gate.
- [x] T-011: Run projected-UV residual `open_drawer` policy gate.
  - Use `DEPTH_AUX_TARGET=projected_keypose_uv`.
  - Use `DEPTH_AUX_OUTPUT_DIM=4`.
  - Feed the predicted projected UV bottleneck into the bounded action residual over the RGB-only anchor.
  - Result: no-go. A `5000` step run trained stably, but paired normal-vs-cross action delta stayed exactly `0.0`.
- [x] T-012: Replace projected-UV coordinate bottleneck with full spatial heatmap residual.
  - Result: implemented and verified full projected heatmap logits plus soft-argmax features in the action residual.
  - Outcome: no-go, because same observation with normal vs cross-sample depth still produced identical actions.
- [x] T-013: Train the full heatmap policy gate on `open_drawer`.
  - Use `DEPTH_AUX_TARGET=projected_keypose_heatmap`.
  - Use `DEPTH_AUX_OUTPUT_DIM=512`, `DEPTH_AUX_HEATMAP_SIZE=16`, `DEPTH_AUX_HEATMAP_SIGMA=1.5`.
  - Result: no-go. Paired normal-vs-cross diagnostic stayed exactly zero, so no rollout was run.
- [x] T-014: Prototype a point-action attention gate.
  - Candidate A: Act3D/PerAct-style voxel or point query map that selects a 3D translation target before action decoding.
  - Candidate B: point-action attention over dense point tokens, with the attended 3D point used directly as a translation residual target.
  - Result: no-go. The `open_drawer` point-action gate trained to `5000` steps, but paired normal-vs-cross-sample action delta remained exactly `0.0`.
- [x] T-015: Prototype a coarse 3D action map or waypoint action head.
  - Candidate A: Act3D/PerAct-style voxel or point query map that selects a 3D translation target before action decoding.
  - Candidate B: waypoint action head where the selected 3D point is the primary translation target, not a bounded residual.
  - Candidate C: diffusion-style action head conditioned on dense 3D features.
  - Result: primary waypoint-action produced a tiny nonzero normal-vs-cross action delta, but strict metrics were nearly tied and normal/null/cross rollout all failed with `InvalidActionError`.
- [x] T-016: Wire long-horizon RLBench xyz auxiliary targets.
  - Targets: `future_pose_xyz`, `final_pose_xyz`, `farthest_future_pose_xyz`.
  - Verify: real `open_drawer` HDF5 label smoke produced finite `(3,)` labels; runner dry-run passed `--aux_future_horizon 10`; one-step `farthest_future_pose_xyz` training smoke produced `(1, 3)` prediction/label shapes.
- [x] T-017: Run small `farthest_future_pose_xyz` waypoint gate.
  - Result: no-go. `paired_pred_xyz_l2=1.66e-04`; strict normal/cross `xyz_rmse` was tied (`0.003190` vs `0.003189`).
- [x] T-018: Run 5000-step `farthest_future_pose_xyz` waypoint confirmation gate.
  - Result: no-go. `paired_pred_xyz_l2=1.00e-04`; strict normal/cross `xyz_rmse` was effectively tied, with cross-sample slightly better (`0.003167` vs `0.003162`).
- [x] T-019: Build a ManiSkill3 adapter smoke.
  - Verified: isolated ManiSkill3 venv installed; `PushCube-v1` state and pointcloud random HDF5 smoke exported and validated.
- [x] T-020: Train a lightweight point-cloud action decoder pilot.
  - Candidate: DP3-style compact point-cloud encoder + proprio/language embedding + action chunk decoder.
  - Result: offline pilot completed on ManiSkill3 `PushCube-v1` and `PickCube-v1`. `PickCube-v1` produced much larger paired normal-vs-cross action deltas (`~0.0223`) than RLBench waypoint recipe (`1e-4`), motivating closed-loop scaling.
- [x] T-021: Convert the ManiSkill3 point-cloud decoder pilot into closed-loop diagnostics with matched null/cross controls.
  - Current smoke: tiny single-step PointNet BC failed PickCube rollout (`0/3` for normal/null/cross and proprio baseline). Minimal 8-step action chunk also failed (`0/3` for normal/null/cross).
  - Goal-conditioned PointNet failed strict gate and closed-loop (`0/3` for normal/null/cross).
  - Object-centric feature MLP strongly passed offline gate (`paired normal-vs-cross L2=1.314387`) but failed closed-loop (`0/10` for normal/null/cross).
  - Positive diagnostic: explicit pointcloud geometry controller solves PickCube normal `7/10` at 100 steps and `8/10` at 150 steps, while null/cross controls are far behind.
  - Learned positive diagnostic, stage 1: phase-conditioned geometry-teacher distillation solves normal `17/30`, while null/cross_demo remain `0/30`.
  - Stricter control: phase alone is not enough; null geometry + normal phase and cross geometry + normal phase are both `0/10`.
  - Later update: raw cropped pointcloud was scaled to `100` successful teacher episodes; single-step h256/10k reached normal `20/60`, eval-time null/cross `1/60`, matched sampled-RGB-only/null train baselines `1/60` / `3/60`. This supports a ManiSkill pointcloud gain claim, but not an OpenVLA end-to-end claim.
- [x] T-022: Replace hand-written PickCube phase with learned phase prediction.
  - Phase classifier validation accuracy `96.1%`, macro accuracy `91.6%`.
  - Learned-phase rollout: normal `19/30`, null `0/30`, cross_demo `0/30`.
  - Learned-phase disentanglement: null geometry + learned normal phase, cross geometry + learned normal phase, and normal geometry + learned null phase all `0/10`.
  - Interpretation: phase is no longer only a hand-written signal, but the policy still uses segmentation-derived object features rather than raw RGB-D/OpenVLA tokens.
- [x] T-023: Replace object-feature cube input with raw cropped pointcloud plus cube auxiliary supervision.
  - Data: geometry-teacher pointcloud rollouts, `z>0.02` point sampling, 30 successful episodes, 2366 transitions.
  - Offline gate passed: normal action RMSE `0.103`, null `0.184`, cross_sample `0.215`; normal cube RMSE `0.009m`, null/cross about `0.075m`; paired normal-vs-cross action L2 `0.215`.
  - Closed-loop 30 episodes: normal `2/30`, null `0/30`, cross_demo `0/30`.
  - Interpretation: raw pointcloud now has a weak positive rollout signal, but simple PointNet + single-step action remains far below object-feature learned-phase `19/30`.
- [x] T-024: Use learned raw-pointcloud cube predictions inside the fixed geometry controller.
  - 30-episode closed-loop: normal `22/30`, null `1/30`, cross_demo `0/30`.
  - Interpretation: raw pointcloud perception is strong enough for control; action/temporal decoder is now the main bottleneck.
- [x] T-025: Scale raw cropped-pointcloud teacher distillation and compare single-step vs action-chunk decoders.
  - Data: `100` successful geometry-teacher episodes, `103` attempts, `8388` transitions.
  - Single-step h256/10k offline gate passed: normal action RMSE `0.067`, null `0.210`, cross_sample `0.131`; paired normal-vs-cross L2 `0.120`.
  - Closed-loop single-step, two eval seeds: normal `20/60`, null `1/60`, cross_demo `1/60`.
  - Matched no-depth train baselines with the same data/model/eval seeds: sampled-RGB-only `1/60`, null/proprio `3/60`.
  - Action-chunk h8/h256/10k offline gate passed: paired normal-vs-cross step L2 `0.188`.
  - Closed-loop action-chunk did not improve over single-step: normal `2/30`, null `1/30`, cross_demo `1/30` for the best exec=1 setting.
  - Interpretation: scaling teacher data produces the strongest raw-pointcloud learned-action result so far; shallow chunking is not the answer.

## Depth-Action Coupling

- [x] Stop treating reach-only success as depth evidence; it is now a sanity/regression gate.
- [x] Add a spatial-action target where corrupted depth cannot solve the auxiliary task and where its output structurally determines translation.
- [x] Add action residual conditioning directly from absolute keypose auxiliary output.
- [x] Add paired normal-vs-cross-sample action-delta diagnostic so the causal gate can detect "all modes identical" before rollout.
- [x] Use the paired diagnostic to reject keypose-conditioned residual as sufficient: normal and cross-sample action predictions were identical.
- [x] Implement PointVLA-style RGB anchor protection as the default safe recipe: low gate init, frozen RGB/LoRA/proprio/action-head base, train depth/query/residual only.
- [x] Add optional hidden/action residual clamp so depth can only provide a bounded correction to a working RGB action.
- [x] Prototype BridgeVLA-style projected multi-view keypose heatmap as an offline gate.
- [x] Add minimal projected-UV spatial bottleneck that can be wired into action residual.
- [x] Train projected-UV residual and reject it if paired normal-vs-cross action delta remains zero.
- [x] Wire full heatmap logits or soft-argmax into final action formation; projected UV alone was too weak.
- [x] Run the full heatmap `open_drawer` gate and reject it because paired normal-vs-cross action delta remained zero.
- [x] Prototype bounded point-action attention; rejected because normal-vs-cross-sample action delta remained exactly zero.
- [x] Prototype Act3D-style coarse-to-fine 3D action map or waypoint head where 3D selection is the primary action output.
- [x] Add a metric that measures prediction delta between normal/null/shuffle depth for the same observation.
  - Update: `diagnose_policy_actions.py` now records selected depth point and waypoint-action debug metrics, including paired normal-vs-cross geometry deltas when `DIAG_COMPARE_DEPTH_MODE` is set.
- [x] Add a 3D action-map candidate coverage probe before another full policy run.
- [x] Add rollout/eval depth mode `shuffle_samples` or `replace_from_other_episode` for dense point tokens.
- [x] Try RGB anchor protection: initialize depth fusion gate near `0`, freeze LoRA/RGB backbone, train only depth/query/residual modules first.
- [x] Add a residual clamp option so normal depth can improve action but cannot dominate successful RGB policy.

## Dataset Scaling

- [ ] Generate or validate stable6 `10 demos/task`. Defer until a new point-cloud action decoder passes a small normal/null/cross causal gate.
- [ ] Train matched stable6 RGB-only and RGB-D on `10 demos/task`. Defer; current RLBench waypoint/residual recipe is no-go.
- [ ] Expand to PerAct/RVT-style 18-task set only after a ManiSkill/RLBench closed-loop pilot passes normal/null/cross and matched-baseline gates.
- [x] Start ManiSkill3 adapter as the next positive-result route, because RLBench residual/waypoint gates have failed.
- [x] Do not scale ManiSkill3 beyond smoke until normal beats null/cross-sample in offline action-delta and action-loss diagnostics.
  - Result: small offline gates passed enough to justify next closed-loop pilot, especially `PickCube-v1`; still do not claim RGB-D improvement without rollout.
- [x] Redesign labels toward object/contact-conditioned 3D targets before scaling; current next-pose/keypose labels are too close to EE and encourage proprio shortcuts.
  - Update: added `visible_object_point_xyz` and `visible_object_rel_xyz` as OpenVLA/RLBench geometry bottleneck targets.
- [ ] Train the first OpenVLA `visible_object_point_xyz` gate on `open_drawer` and reject it unless selected point / waypoint / action paired deltas pass.
- [x] Train a final `farthest_future_pose_xyz` waypoint/action-map run only if there is enough time and disk; do not claim success unless normal beats null/cross and RGB-only.

## Reproducibility

- [ ] Add config files for reach-only RGB/RGB-D experiments.
- [x] Add exact reproduction commands to a top-level `EXPERIMENTS.md`.
- [x] Add a script to collect RLBench eval and diagnostic JSON into one table.

## Paper / Reporting

- [x] Write limitations: LIBERO saturation, weak depth-action coupling, RLBench policy drift.
- [x] Keep negative results as structured evidence, not vague "failed" notes.
- [x] Add the `open_drawer` keypose-residual no-go as a concrete negative result: auxiliary keypose coupling alone did not make depth causally affect action.
- [x] Add the `open_drawer` projected-UV residual no-go as a concrete negative result: auxiliary UV supervision alone did not make depth causally affect action.
- [x] Add the `open_drawer` point-action residual no-go as a concrete negative result: point selection plus bounded residual still did not make depth causally affect action.
- [x] Add the `open_drawer` primary waypoint-action no-go as the final concrete negative result: making selected 3D point the primary xyz action created only tiny causal deltas and failed closed-loop.
- [x] Add the `open_drawer` farthest-future waypoint no-go as the final confirmation: changing to long-horizon target and training `5000` steps still did not create normal-depth causal action gain.
