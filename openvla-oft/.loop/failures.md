# Failure Log

## F-001: LIBERO clean is saturated for proving depth benefit

Date: 2026-07-03

Symptom:
- RGB-only reaches clean trained-task saturation.
- Depth variants do not show stable rollout improvement over RGB-only.

Possible reasons:
- Benchmark ceiling effect.
- Strong RGB-language/proprio shortcut.
- Shallow depth fusion is optional and can be ignored.

Evidence:
- Historical local documents and eval logs archived in `docs_archive_20260616_restart/`.

Next action:
- Keep LIBERO only for sanity checks and historical comparison.
- Move main causal evidence to RLBench.

## F-002: RLBench stable6 3-demo policy rollout fails even though demo replay works

Date: 2026-07-03

Symptom:
- Demo next-absolute-pose replay succeeds with max_steps 200.
- RGB-only and RGB-D normal stable6 policy eval both produce `0/6` at horizon 150.
- Failures are mainly `InvalidActionError` after multiple steps.

Possible reasons:
- Closed-loop drift from small single-step direction errors.
- `3 demos/task + 2000 steps` is undertrained.
- Action adapter may need single-task overfit validation.

Evidence:
- `experiments/logs/rlbench_action_replay_next_abs_planning_200.json`
- `experiments/logs/rlbench_eval_results_h150/rgb_only.json`
- `experiments/logs/rlbench_eval_results_h150/rgbd_normal.json`

Next action:
- Run reach-only overfit before scaling.

## F-003: RGB-D action prediction does not causally depend on depth content yet

Date: 2026-07-03

Symptom:
- Balanced offline action diagnostic:
  - RGB-D normal xyz RMSE `0.00514`
  - RGB-D null xyz RMSE `0.00515`
  - RGB-D shuffle xyz RMSE `0.00514`
- Normal/null/shuffle are nearly identical.

Possible reasons:
- Depth branch helps absolute keypose probe but not the VLA action head.
- Behavior cloning loss can still use RGB/proprio shortcut.
- Depth fusion remains optional for action prediction.

Evidence:
- `experiments/logs/rlbench_policy_action_diag_rgbd_normal.json`
- `experiments/logs/rlbench_policy_action_diag_rgbd_null.json`
- `experiments/logs/rlbench_policy_action_diag_rgbd_shuffle.json`

Next action:
- Strengthen depth-action coupling after reach-only action adapter is validated.

## F-004: Reach-only RGB-D normal depth fails closed-loop while RGB-only and RGB-D null succeed

Date: 2026-07-03

Symptom:
- RGB-only `reach_target` overfit succeeds for `MAX_DELTA_XYZ=0.03/0.05/0.08`, all `1/1`, length `29`.
- RGB-D dense/keypose with `DEPTH_AUX_SPATIAL_LOSS_WEIGHT=0.2` trains to `5000` steps.
- RGB-D normal fails closed-loop for `MAX_DELTA_XYZ=0.03/0.05/0.08`, all `0/1`, length `150`.
- RGB-D null succeeds at `MAX_DELTA_XYZ=0.05`, `1/1`, length `31`.
- RGB-D shuffle fails at `MAX_DELTA_XYZ=0.05`, `0/1`, length `150`.

Possible reasons:
- Normal depth context perturbs the successful RGB action policy instead of improving it.
- The action head uses depth in a way that changes closed-loop behavior but is not aligned with success.
- Pixel shuffle is too weak for dense point tokens; it preserves point-set/statistical structure and behaves like normal in offline diagnostic.
- Offline single-step action RMSE is not sufficient to guarantee closed-loop success.

Evidence:
- `experiments/logs/rlbench_eval_reach_overfit_rgb_h150_d003/rgb_only.json`
- `experiments/logs/rlbench_eval_reach_overfit_rgb_h150_d005/rgb_only.json`
- `experiments/logs/rlbench_eval_reach_overfit_rgb_h150_d008/rgb_only.json`
- `experiments/logs/rlbench_eval_reach_overfit_rgbd_aux02_h150_d003/rgbd_normal.json`
- `experiments/logs/rlbench_eval_reach_overfit_rgbd_aux02_h150_d005/rgbd_normal.json`
- `experiments/logs/rlbench_eval_reach_overfit_rgbd_aux02_h150_d005/rgbd_null.json`
- `experiments/logs/rlbench_eval_reach_overfit_rgbd_aux02_h150_d005/rgbd_shuffle.json`
- `experiments/logs/rlbench_eval_reach_overfit_rgbd_aux02_h150_d008/rgbd_normal.json`

Next action:
- Do not scale stable6 yet.
- Protect the RGB anchor and make depth residual/gating safer.
- Add cross-sample depth corruption for dense-point causal tests.
# Failures

## F-2026-07-03-safe-rgbd-without-rgb-anchor

Date: 2026-07-03

Context:
- Tried reach-only safe RGB-D training with low gate, hidden/action clamp, frozen VLA/proprio/action-head base.
- Command used `FREEZE_VLA_LORA=True`, `FREEZE_PROPRIO_PROJECTOR=True`, `FREEZE_ACTION_HEAD_BASE=True`.
- The run did **not** set `resume_components_from`, so the action head/proprio/Lora anchor were not loaded from the successful RGB-only reach checkpoint before freezing.

Result:
- `rgbd_normal`: `0/1`, length `50`, `InvalidActionError`.
- `rgbd_null`: `0/1`, length `60`, `InvalidActionError`.
- `rgbd_cross_sample`: `0/1`, length `50`, `InvalidActionError`.

Why this run is invalid:
- The intended experiment was "protect the working RGB action expert".
- The actual experiment froze a newly initialized action-head base, then trained only depth/query-related modules.
- Therefore failure does not prove low-gate/clamped RGB-D fusion is bad; it proves the runner must load the RGB anchor before freezing.

Fix:
- Added `RESUME_COMPONENTS_FROM` and `RESUME_STEP` to `experiments/robot/rlbench/run_rlbench_rgbd_stage.sh`.
- Added `resume_components_from` and `resume_step` to `depthvla_config.json`.
- Next safe run must use the successful RGB-only reach checkpoint as `RESUME_COMPONENTS_FROM`.

## F-2026-07-04-safe-rgbanchor-depth-ignored

Date: 2026-07-04 UTC

Context:
- Re-ran reach-only safe RGB-D training with the successful RGB-only reach checkpoint loaded through `RESUME_COMPONENTS_FROM`.
- Used low gate, hidden/action residual clamps, depth dropout, and frozen VLA/proprio/action-head base.

Result:
- Closed-loop reach rollout at `MAX_DELTA_XYZ=0.05`:
  - normal: `1/1`, length `29`
  - null: `1/1`, length `29`
  - cross_sample: `1/1`, length `29`
- Offline policy diagnostic:
  - normal/null/cross_sample all `xyz_rmse=0.001700`
  - normal/null/cross_sample all `xyz_direction_cosine=0.97756`

Interpretation:
- The repair succeeded at protecting the RGB policy: normal depth no longer breaks reach.
- It failed the depth causal gate: normal, null, and cross-sample depth produce indistinguishable actions and rollout traces.
- `reach_target` is now too easy and too RGB/proprio solvable to prove metric depth usage.

Next action:
- Treat reach as a regression/sanity check only.
- Move the next causal gate to 3D-sensitive tasks such as `open_drawer`, `turn_tap`, `slide_block_to_target`, `pick_up_cup`, or a ManiSkill pilot with strong pose/view/height variation.
- Add explicit spatial action grounding: keypose-conditioned residual, projected heatmap, or coarse-to-fine 3D action map.

## F-2026-07-04-open-drawer-safe-residual-no-go

Date: 2026-07-04 UTC

Context:
- Switched from easy `reach_target` to 3D/contact task `open_drawer`.
- Offline dense-depth keypose probe passed strongly:
  - normal xyz RMSE `0.0705`
  - null xyz RMSE `0.1420`
  - shuffle xyz RMSE `0.1466`
- RGB-only baseline trained for `5000` steps and did not saturate: `0/1`, length `200`, no invalid action.
- Safe RGB-D trained from the RGB-only anchor with low gate, clamps, frozen RGB/proprio/action-head base, and absolute-keypose auxiliary loss.

Result:
- Rollout, `open_drawer`, horizon `200`, `MAX_DELTA_XYZ=0.05`:
  - RGB-only: `0/1`, length `200`
  - RGB-D normal: `0/1`, length `200`
  - RGB-D null: `0/1`, length `200`
  - RGB-D cross_sample: `0/1`, length `200`
- Offline action diagnostic:
  - normal xyz RMSE `0.001336`, cosine `0.78733`
  - null xyz RMSE `0.001357`, cosine `0.78683`
  - cross_sample xyz RMSE `0.001336`, cosine `0.78733`

Interpretation:
- `open_drawer` is a better benchmark than reach: RGB-only does not saturate and depth has offline keypose signal.
- Shallow safe residual still fails to convert depth signal into causal closed-loop action improvement.
- Since normal and cross-sample are identical, the action head is still not using true depth geometry.

Next action:
- Stop simple low-gate residual sweeps.
- Implement explicit spatial action grounding where the auxiliary spatial prediction participates in action, not just as a side loss.
- Candidate designs: keypose-conditioned residual, projected multi-view heatmap, or coarse-to-fine 3D action map.

## F-2026-07-04-keypose-residual-still-ignored

Date: 2026-07-04 UTC

Context:
- Added keypose-conditioned action residual to make the absolute keypose auxiliary output affect the final delta action.
- Trained `open_drawer` from the RGB-only anchor for `5000` steps with frozen RGB/proprio/action-head base and bounded residuals.

Result:
- Training was stable and saved a `5000` step checkpoint.
- Paired normal-vs-cross-sample diagnostic:
  - `paired_pred_l1=0.0`
  - `paired_pred_rmse=0.0`
  - `paired_pred_xyz_l2=0.0`
- Strict diagnostic:
  - normal RMSE `0.0137849`
  - null RMSE `0.0137148`
  - cross-sample RMSE `0.0137849`

Interpretation:
- Scalar keypose-conditioned residual was still optional enough for the action path to ignore true depth content.
- Good auxiliary keypose signal is not sufficient unless the spatial output is structurally tied to action formation.

Next action:
- Do not scale this checkpoint or run long rollout.
- Use projected heatmap or 3D action-map output as the next action-grounded bottleneck.

## F-2026-07-04-projected-uv-residual-still-ignored

Date: 2026-07-04 UTC

Context:
- A projected-keypose heatmap probe showed real depth sensitivity on `open_drawer` and stable6.
- Added a minimal trainable policy bottleneck, `DEPTH_AUX_TARGET=projected_keypose_uv`, with normalized `[agent_u, agent_v, wrist_u, wrist_v]` labels.
- Trained `open_drawer` from the RGB-only anchor for `5000` steps with frozen RGB/proprio/action-head base modules and bounded keypose/action residuals.

Result:
- Training was stable and saved a `5000` step checkpoint.
- Paired normal-vs-cross-sample diagnostic:
  - `paired_pred_l1=0.0`
  - `paired_pred_rmse=0.0`
  - `paired_pred_xyz_l2=0.0`
  - `paired_pred_rpy_l2=0.0`
  - `paired_pred_gripper_abs=0.0`
- The diagnostic output is recorded at `experiments/logs/rlbench_policy_action_diag_rgbd_normal.json`.

Interpretation:
- Projected heatmap supervision is causally sensitive to real depth in a small probe, but compressing it into four UV scalars and routing it through the bounded residual is still too optional for the policy.
- The action prediction path continues to ignore true depth content when RGB/proprio can explain the behavior-cloning loss.
- This rejects projected-UV coordinates as the next scalable policy bottleneck.

Next action:
- Do not run rollout or scale this checkpoint.
- Delete the no-go checkpoint after recording results.
- Wire full heatmap logits/soft-argmax features or a coarse 3D action map directly into final action formation, then require nonzero paired normal-vs-cross action deltas before rollout.

## F-2026-07-04-maniskill-goal-conditioned-pointnet-no-go

Date: 2026-07-04 UTC

Context:
- Added `goal_pos` to the ManiSkill3 PickCube pointcloud/proprio PointNet decoder.
- Motivation: the successful geometry controller uses `goal_pos`, while the earlier learned decoders did not.

Result:
- Offline gate:
  - normal RMSE `0.196182`
  - null RMSE `0.206774`
  - cross_sample RMSE `0.195966`
  - paired normal-vs-cross L2 `0.019890`
- Closed-loop smoke, 3 episodes:
  - normal `0/3`, mean reward `5.97`
  - null `0/3`, mean reward `5.53`
  - cross_demo `0/3`, mean reward `6.61`

Interpretation:
- Adding goal conditioning helps relative to null, but normal still does not beat cross_sample.
- Goal conditioning alone does not solve real-geometry closed-loop control.

Next action:
- Stop tiny PointNet BC variants as the main route.
- Use object-centric geometry features or a temporal policy instead.

## F-2026-07-04-maniskill-object-feature-bc-no-go

Date: 2026-07-04 UTC

Context:
- Added an object-centric feature MLP using pointcloud segmentation cube center, `tcp_pose`, `goal_pos`, `is_grasped`, relative 3D vectors, and proprio.
- Motivation: test whether explicitly extracting the object-conditioned geometry can force action prediction to depend on real pointcloud content.

Result:
- Offline gate strongly passed:
  - normal RMSE `0.130584`
  - null RMSE `51700.929688`
  - cross_sample RMSE `0.990587`
  - paired normal-vs-cross L2 `1.314387`
- Closed-loop smoke, 10 episodes:
  - normal `0/10`, mean reward `4.83`
  - null `0/10`, mean reward `0.51`
  - cross_demo `0/10`, mean reward `5.25`

Interpretation:
- The policy now causally sees geometry in offline action prediction.
- Single-step behavior cloning still fails closed-loop, so the remaining gap is temporal control / compounding error, not raw geometry extraction.
- The paired geometry-controller result proves the same task is solvable when a proper geometry-to-action state machine is supplied.

Next action:
- Move to DP3/diffusion, ACT-style temporal aggregation, DAgger-style correction, or distillation from the geometry controller.
- Do not claim final RGB-D success from this model.

## F-2026-07-04-maniskill-teacher-distillation-without-phase-no-go

Date: 2026-07-04 UTC

Context:
- Collected 30 successful geometry-controller teacher episodes from ManiSkill3 PickCube.
- Trained an object-feature MLP on teacher actions without explicit phase input.

Result:
- Dataset: `30` successful episodes, `2538` transitions.
- Validation raw RMSE `0.100967`.
- Closed-loop, 150 steps, 10 episodes:
  - normal `0/10`, mean reward `54.56`
  - null `0/10`, mean reward `3.67`
  - cross_demo `0/10`, mean reward `12.83`
- 300-step normal also stayed `0/10`, mean reward `112.72`.
- Debug showed the policy often grasped the cube but kept it lifted away from the goal, failing to transition reliably into move-goal.

Interpretation:
- Teacher data and object-centric geometry improved behavior substantially, but a memoryless single-step MLP cannot infer the hidden phase robustly enough.
- This confirms the next bottleneck is temporal state / phase inference.

Next action:
- Use phase-conditioned distillation as a diagnostic.
- For a real method, replace hand-supplied phase with ACT/DP3/recurrent/diffusion temporal state learning.

## F-2026-07-04-projected-heatmap-policy-still-ignored

Date: 2026-07-04 UTC

Context:
- Implemented `DEPTH_AUX_TARGET=projected_keypose_heatmap` with two-view `16x16` Gaussian labels from projected `rlbench_keypose_action[:3]`.
- The action residual received both full heatmap logits (`512` dims) and soft-argmax UV coordinates.
- Trained `open_drawer` from the RGB-only anchor for `5000` steps with frozen RGB/proprio/action-head base modules and bounded depth/keypose/action residuals.

Result:
- Training was stable and saved a `5000` step checkpoint.
- Paired normal-vs-cross-sample diagnostic:
  - `paired_pred_l1=0.0`
  - `paired_pred_rmse=0.0`
  - `paired_pred_xyz_l2=0.0`
  - `paired_pred_rpy_l2=0.0`
  - `paired_pred_gripper_abs=0.0`
- Single-step action RMSE stayed small (`rmse=0.0142`, `xyz_rmse=0.00125`), but the paired depth intervention produced no action change.

Interpretation:
- Full projected heatmap supervision is still optional under the current bounded residual design.
- Feeding heatmap logits into a residual MLP is not enough to force the policy to use true metric depth geometry.
- The failure is stronger than the projected-UV no-go: even preserving the spatial distribution did not create depth-action coupling.

Next action:
- Do not run rollout or scale this checkpoint.
- Delete the no-go checkpoint after recording results.
- Move to a harder structural bottleneck where the selected 3D waypoint/action is the primary translation output, not a bounded residual.

## F-2026-07-04-point-action-gate-still-ignored

Date: 2026-07-04 UTC

Context:
- Implemented a stronger point-action gate over dense point tokens.
- The branch scores sampled 3D points using an action/language-conditioned query, predicts `point_keypose_xyz`, and can add a bounded point-selected translation residual.
- Fixed a critical RLBench intrinsics issue first: negative focal lengths were previously clamped to `1e-6`, exploding point clouds to `1e8` scale. After the fix, point clouds were meter-scale and a real `open_drawer` one-step point aux smoke loss was about `0.0327`.
- Trained `open_drawer` from the RGB-only anchor for `5000` steps with `DEPTH_AUX_TARGET=point_keypose_xyz`, `DEPTH_POINT_ACTION_WEIGHT=1.0`, and `DEPTH_POINT_ACTION_CLIP=0.02`.

Result:
- Training was stable and saved a `5000` step checkpoint.
- Paired normal-vs-cross-sample diagnostic:
  - `paired_pred_l1=0.0`
  - `paired_pred_rmse=0.0`
  - `paired_pred_xyz_l2=0.0`
  - `paired_pred_rpy_l2=0.0`
  - `paired_pred_gripper_abs=0.0`
- Single-step action error was small (`rmse=0.022144`, `xyz_rmse=0.001285`), but replacing depth with cross-sample depth produced no action change.

Interpretation:
- Even a point-selection auxiliary target plus bounded residual can remain optional when the RGB/proprio action expert already explains the behavior-cloning loss.
- This rejects bounded point-action residuals as the next scalable policy bottleneck.
- The next method must make the selected 3D action or waypoint the primary action output, not just a residual feature added to the existing action head.

Next action:
- Do not run rollout or scale this checkpoint.
- Delete the no-go checkpoint after recording results.
- Move to an Act3D/PerAct-style coarse 3D action map, point-voxel action classification, or waypoint/diffusion action head with mandatory normal-vs-cross-sample action-delta gating before rollout.

## F-2026-07-04-primary-waypoint-action-no-go

Date: 2026-07-04 UTC

Context:
- Implemented a final primary waypoint-action attempt for `open_drawer`.
- The model reused dense point tokens and `point_keypose_xyz` supervision, but set `DEPTH_WAYPOINT_ACTION_WEIGHT=1.0` so the selected 3D point directly controlled first-step xyz instead of acting as an optional residual.
- Used `DEPTH_WAYPOINT_ACTION_CLIP=0.02`, frozen RGB/proprio/action-head base modules, and resumed from the matched RGB-only `open_drawer` anchor.

Result:
- Training completed to `5000` steps.
- Paired normal-vs-cross-sample diagnostic was finally nonzero, but tiny:
  - `paired_pred_l1=5.90e-05`
  - `paired_pred_rmse=1.15e-04`
  - `paired_pred_xyz_l2=3.05e-04`
- Strict diagnostic showed only negligible normal advantage:
  - normal `xyz_rmse=0.003178`
  - null `xyz_rmse=0.003210`
  - cross-sample `xyz_rmse=0.003226`
- Closed-loop rollout failed for all depth modes:
  - normal `0/1`, length `11`, `InvalidActionError`
  - null `0/1`, length `10`, `InvalidActionError`
  - cross-sample `0/1`, length `11`, `InvalidActionError`

Interpretation:
- Making the selected 3D point the primary xyz action created a measurable but very weak causal dependency on depth.
- The dependency was not strong or stable enough to improve action prediction meaningfully or survive closed-loop execution.
- The final result remains a negative result: current OpenVLA-OFT RGB-D adapters did not convert metric depth signal into robust rollout improvement.

Next action:
- Do not claim RGB-D improvement from this run.
- Keep the JSON logs and code as reproducibility evidence.
- Delete the no-go checkpoint to save disk; keep only useful RGB-only anchors and compact result docs.

## F-2026-07-04-farthest-future-waypoint-still-no-go

Date: 2026-07-04 UTC

Context:
- The 3D action-map feasibility probe showed that short-horizon keypose/next-pose labels have a strong EE/proprio shortcut.
- Wired long-horizon targets into training: `future_pose_xyz`, `final_pose_xyz`, and `farthest_future_pose_xyz`.
- Ran a small `open_drawer` 500-step gate with `DEPTH_AUX_TARGET=farthest_future_pose_xyz`, `DEPTH_WAYPOINT_ACTION_WEIGHT=1.0`, `DEPTH_WAYPOINT_ACTION_CLIP=0.02`, and dense point tokens at `256` points/view.
- Because the 500-step gate could be dismissed as undertraining, also ran a 5000-step confirmation gate with the same target and waypoint-action recipe.

Result:
- Training completed and saved a 500-step checkpoint.
- Paired normal-vs-cross-sample diagnostic:
  - `paired_pred_l1=2.65e-05`
  - `paired_pred_rmse=6.26e-05`
  - `paired_pred_xyz_l2=1.66e-04`
- Strict diagnostic remained tied:
  - normal `xyz_rmse=0.003190`
  - null `xyz_rmse=0.003210`
  - cross-sample `xyz_rmse=0.003189`
- The 5000-step confirmation also failed:
  - paired normal-vs-cross `paired_pred_l1=1.58e-05`
  - paired normal-vs-cross `paired_pred_rmse=3.79e-05`
  - paired normal-vs-cross `paired_pred_xyz_l2=1.00e-04`
  - strict normal `xyz_rmse=0.003167`
  - strict null `xyz_rmse=0.003210`
  - strict cross-sample `xyz_rmse=0.003162`

Interpretation:
- Simply changing the auxiliary target horizon does not solve the action coupling problem.
- The paired action delta is even smaller than the previous primary waypoint-action run, and it shrinks further after 5000 steps.
- Cross-sample depth slightly beats normal depth in the 5000-step strict diagnostic, so there is no evidence of causal normal-depth advantage.
- The current OpenVLA-OFT action path still compresses depth into a weak perturbation instead of making 3D geometry a necessary action variable.

Next action:
- Do not run rollout or scale this recipe.
- Delete the no-go checkpoints after recording logs.
- Move to a real 3D action map, object/contact-conditioned waypoint target, or DP3-style point-cloud action decoder before further scaling.
