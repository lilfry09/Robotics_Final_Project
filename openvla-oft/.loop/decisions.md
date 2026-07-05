# Decisions

## D-001: Do not use clean LIBERO as the main RGB-D evidence

Date: 2026-07-03

Reason:
- RGB-only OpenVLA-OFT is already near saturation on clean LIBERO.
- A saturated benchmark hides the marginal benefit of metric depth.
- LIBERO remains useful for sanity checks and historical comparison.

Evidence:
- Local documents record RGB-only clean trained tasks at `15/15`.
- `NEXT_RGBD_BENCHMARK_PLAN.md` and `RGBD_DATASET_ACTIONSPACE_RESEARCH.md`.

Status:
Accepted

## D-2026-07-04-learned-phase-pickcube-is-positive-diagnostic-not-final-vla

Date: 2026-07-04 UTC

Reason:
- The hand-written phase-conditioned teacher policy already showed normal geometry can drive learned actions: normal `17/30`, null `0/30`, cross_demo `0/30`.
- A follow-up phase classifier learned phase from object-centric 3D features with validation accuracy `96.1%` and macro accuracy `91.6%`.
- Replacing the hand-written phase with the learned classifier during rollout improved/confirmed the positive result: normal `19/30`, null `0/30`, cross_demo `0/30`.
- Disentanglement controls stayed strict: null geometry + learned normal phase, cross geometry + learned normal phase, and normal geometry + learned null phase are all `0/10`.

Decision:
- Treat ManiSkill3 PickCube learned-phase object-feature distillation as the strongest positive diagnostic so far.
- Do not claim final OpenVLA RGB-D success, because the policy still uses segmentation-derived object features rather than raw RGB-D/OpenVLA tokens.
- Next implementation should move this object-centric geometry/phase result into a raw pointcloud/RGB-D temporal policy: ACT-style temporal aggregation, DP3/diffusion action decoding, recurrent state, or 3D action-map classification.

Status:
Accepted

## D-2026-07-04-raw-cropped-pointcloud-needs-object-centric-auxiliary

Date: 2026-07-04 UTC

Reason:
- A raw full-scene pointcloud teacher policy with phase/action heads learned phase but failed the strict normal-vs-cross action gate.
- Adding a cube-center auxiliary head without changing point sampling still failed: normal and cross cube RMSE were both about `0.025m`, suggesting the model was mostly using task/proprio shortcuts.
- Cropping low table points with `z>0.02` made the object geometry learnable: normal cube RMSE became `0.009m`, cross/null about `0.075m`, and paired normal-vs-cross action L2 rose to `0.215`.
- Closed-loop improved from all-zero to weak positive: normal `2/30`, null `0/30`, cross_demo `0/30`.

Decision:
- Keep the raw cropped-pointcloud result as a weak positive bridge between object-feature distillation and raw RGB-D/pointcloud policy.
- Do not claim final success from it because normal success is only `2/30`.
- Future raw pointcloud policy should keep object-centric auxiliary supervision and use a stronger encoder/temporal decoder, such as PointNet++/Transformer point queries plus ACT or DP3-style diffusion action chunks.

Status:
Accepted

## D-2026-07-04-action-decoder-is-now-the-main-bottleneck

Date: 2026-07-04 UTC

Reason:
- The raw cropped-pointcloud model with cube auxiliary supervision predicts cube center accurately offline: normal cube RMSE `0.009m`, null/cross about `0.075m`.
- Its learned single-step action head remains weak in closed loop: normal `2/30`, null `0/30`, cross_demo `0/30`.
- Feeding the same learned cube prediction into the fixed geometry controller solves normal `22/30`, while null is `1/30` and cross_demo `0/30`.

Decision:
- Treat raw pointcloud perception as solved enough for the current PickCube diagnostic.
- The next serious model change should target the action/temporal decoder, not another perception-only probe.
- Preferred routes: ACT-style action chunking with temporal aggregation, DP3/diffusion action sequence decoder conditioned on the object-centric bottleneck, or an Act3D/PerAct-style 3D action map.

Status:
Accepted

## D-2026-07-04-scaled-raw-pointcloud-policy-is-positive-maniskill-evidence

Date: 2026-07-04 UTC

Reason:
- The 30-episode raw cropped-pointcloud teacher policy gave only a weak rollout signal: normal `2/30`, null `0/30`, cross_demo `0/30`.
- Scaling the geometry-teacher data to `100` successful episodes (`8388` transitions) and training a larger h256/10k single-step PointNet policy substantially improved closed-loop success.
- Offline causal gate passed strongly:
  - normal action RMSE `0.067`
  - null action RMSE `0.210`
  - cross_sample action RMSE `0.131`
  - paired normal-vs-cross action L2 `0.120`
- Closed-loop evaluation across two independent 30-episode eval seeds:
  - normal `20/60`
  - null `1/60`
  - cross_demo `1/60`
- Matched train-time no-depth baselines were run with the same 100 teacher episodes, h256/10k capacity, and eval seeds:
  - sampled-RGB-only baseline, xyz zeroed: `1/60`
  - null/proprio baseline: `3/60`
- A scaled h=8 action-chunk model also passed offline gate (`paired_normal_vs_cross_step_l2=0.188`) but did not improve closed-loop success, so the gain is not from naive chunking.

Decision:
- Treat the scaled raw cropped-pointcloud teacher policy as the strongest raw-pointcloud learned action evidence so far.
- It is valid to say that, on ManiSkill3 PickCube, normal pointcloud now produces a clear learned closed-loop gain over both eval-time null/cross controls and train-time no-depth baselines.
- Do not claim this satisfies the original OpenVLA RGB-D > RGB-only objective: the model is a small teacher-distilled PointNet policy, the benchmark is ManiSkill PickCube, and the data come from a geometry teacher.
- The next credible route is to move this raw pointcloud/object-centric bottleneck into a stronger temporal policy, such as DP3/diffusion, ACT-style temporal aggregation, recurrent state, DAgger, or a 3D action-map policy.

Status:
Accepted

## D-015: Primary waypoint action is still not enough for a positive RGB-D claim

Date: 2026-07-04 UTC

Reason:
- The final `open_drawer` waypoint-action run made selected 3D point output the primary first-step xyz action instead of a bounded residual.
- It trained stably for `5000` steps and produced a tiny nonzero paired normal-vs-cross-sample action delta.
- The effect was too small to support a depth-gain claim, strict normal/null/cross metrics were nearly tied, and closed-loop rollout failed for all depth modes with `InvalidActionError`.

Evidence:
- Paired diagnostic:
  - `paired_pred_l1=5.90e-05`
  - `paired_pred_rmse=1.15e-04`
  - `paired_pred_xyz_l2=3.05e-04`
- Strict diagnostic:
  - normal `xyz_rmse=0.003178`
  - null `xyz_rmse=0.003210`
  - cross-sample `xyz_rmse=0.003226`
- Rollout, `open_drawer`, horizon `200`:
  - normal `0/1`, length `11`, `InvalidActionError`
  - null `0/1`, length `10`, `InvalidActionError`
  - cross-sample `0/1`, length `11`, `InvalidActionError`

Rule:
- Do not claim RGB-D beats RGB from this checkpoint.
- Treat this OpenVLA/RLBench waypoint checkpoint as a structured no-go: depth has offline spatial signal, but this action path did not convert it into robust causal control.
- Future work should move beyond quick residual/waypoint patches toward a full action-space redesign or a benchmark/model stack built for 3D action maps.

Status:
Accepted

## D-016: Move the next positive-result route to ManiSkill3 plus a real point-cloud action decoder

Date: 2026-07-04 UTC

Reason:
- RLBench was the right first non-LIBERO gate, but the current OpenVLA-OFT residual/waypoint recipe has now failed multiple hard gates on `open_drawer`.
- The final `farthest_future_pose_xyz` 5000-step confirmation did not create normal-depth causal action gain: paired normal-vs-cross `paired_pred_xyz_l2=1.00e-04`, and strict cross-sample `xyz_rmse=0.003162` was slightly better than normal `0.003167`.
- Continuing to scale stable6/18-task RLBench with the same recipe would mainly spend compute on a disproven coupling mechanism.
- Official ManiSkill3 materials make it a better next data-scaling source because it supports RGB-D/pointcloud observations and high-throughput GPU-parallel visual data collection.
- Method sources such as Act3D and DP3 point to a stronger conclusion: depth must feed a primary 3D action decoder/action map, not an optional residual.
- Local adapter smoke now supports this route:
  - ManiSkill3 installed in `/root/autodl-tmp/envs/maniskill3-venv`
  - `PushCube-v1` state HDF5 smoke passed
  - `PushCube-v1` pointcloud HDF5 smoke passed and validates finite pointcloud/action/proprio fields

Rule:
- RLBench remains a regression and diagnostic benchmark, not the next scaling target for the current recipe.
- Do not start another RLBench stable6/18-task scale-up until a new action decoder passes normal/null/cross action-delta gates.
- Next implementation priority is a lightweight DP3-style point-cloud action decoder or Act3D/PerAct-style 3D action-map head on top of the working ManiSkill3 smoke data path.
- Keep matched RGB-only, normal depth, null depth, and cross-sample depth controls as non-negotiable gates.

Status:
Accepted

## D-017: Early ManiSkill3 point-cloud decoder was promising offline evidence, later superseded

Date: 2026-07-04 UTC

Reason:
- Official ManiSkill3 `PushCube-v1` and `PickCube-v1` demos were replayed into pointcloud observations with `pd_ee_delta_pos` actions.
- A tiny primary point-cloud action decoder produced measurable normal-vs-cross action sensitivity.
- `PushCube-v1` passed the strict offline gate in `3/3` seeds, but the delta is small (`paired normal-vs-cross L2 mean=0.002041`) and matched proprio-only is close.
- `PickCube-v1` produced a much larger action delta (`paired normal-vs-cross L2 mean=0.022263`) and passed `2/3` seeds, but one seed still had cross-sample RMSE slightly below normal.
- A tiny single-step PointNet BC closed-loop smoke was run after this decision:
  - 1200-step pointcloud normal/null/cross_demo all `0/3`.
  - 5000-step pointcloud normal/null/cross_demo all `0/3`.
  - Normal reward was higher than null, but cross_demo reward was also high, so there is no normal-geometry closed-loop gain.
- A minimal 8-step action chunk decoder was also run:
  - Offline paired normal-vs-cross step L2 `0.027554`.
  - Closed-loop normal/null/cross_demo all `0/3`; normal and cross_demo rewards are tied.
- A goal-conditioned PointNet decoder was run:
  - normal RMSE `0.196182`, null `0.206774`, cross_sample `0.195966`.
  - paired normal-vs-cross L2 `0.019890`, but normal did not beat cross_sample.
  - Closed-loop normal/null/cross_demo all `0/3`.
- An object-centric feature MLP was run:
  - It uses pointcloud segmentation cube center plus `tcp_pose`, `goal_pos`, `is_grasped`, relative 3D vectors, and proprio.
  - Offline gate strongly passed: normal RMSE `0.130584`, cross_sample `0.990587`, paired normal-vs-cross L2 `1.314387`.
  - Closed-loop still failed: normal/null/cross_demo all `0/10`.
- A pointcloud geometry controller was run as a feasibility probe:
  - normal pointcloud solved PickCube `7/10`.
  - null pointcloud solved `0/10`.
  - cross-demo pointcloud solved `0/10`.
- A 150-step / last-cube-memory geometry-controller version improved normal to `8/10`, while null stayed `0/10` and cross_demo was `1/10`.
- No matched OpenVLA/RGB-only closed-loop baseline has been run.

Rule:
- This decision describes the early 20-demo stage only.
- It was superseded by `D-2026-07-04-scaled-raw-pointcloud-policy-is-positive-maniskill-evidence`, where the raw cropped pointcloud policy was scaled to `100` successful teacher episodes and achieved normal `20/60` vs eval-time null/cross `1/60` and train-time sampled-RGB-only/null baselines `1/60` / `3/60`.
- The remaining boundary is now narrower: claim ManiSkill3 pointcloud learned-policy gain, but do not claim OpenVLA end-to-end RGB-D > RGB-only.
- The single-step PointNet BC route, shallow chunk route, goal-conditioned PointNet route, and object-feature single-step MLP route are not enough; the next implementation should use DP3/diffusion, ACT-style temporal aggregation, DAgger-style correction, or a stronger 3D action-map policy.
- The object-feature result proves geometry can be made causally visible to action prediction; the geometry-controller result proves it can solve closed-loop. The gap between them is temporal control / compounding error.
- The geometry-controller result is the strongest evidence for the next route: distill or imitate an explicit pointcloud geometry-to-action algorithm, instead of hoping shallow BC discovers it from 20 demos.

Status:
Accepted

## D-018: Treat phase-conditioned geometry-teacher distillation as a learned positive diagnostic, not an OpenVLA result

Date: 2026-07-04 UTC

Reason:
- Geometry controller solved PickCube normal pointcloud but was hand-coded, so it was only a feasibility diagnostic.
- Object-centric single-step BC trained on official demos strongly passed offline normal-vs-cross gates, but failed closed-loop.
- A geometry-teacher dataset was collected from 30 successful controller rollouts.
- Without phase, teacher-distilled object-feature MLP achieved high normal reward but still `0/10` success, and debug showed it grasped the cube but failed to transition reliably to move-goal.
- Adding phase one-hot exposed the missing temporal state while keeping action output learned.

Evidence:
- Teacher phase dataset:
  - `30` successful episodes from `31` attempts.
  - `2335` transitions.
- Phase-conditioned teacher-distilled policy:
  - validation raw RMSE `0.024559`.
  - 10-episode rollout: normal `7/10`, null `0/10`, cross_demo `0/10`.
  - 30-episode rollout: normal `17/30`, null `0/30`, cross_demo `0/30`.
- Phase/geometry disentanglement, same seed, 10 episodes:
  - normal geometry + normal phase: `6/10`, mean reward `39.23`.
  - null geometry + normal phase: `0/10`, mean reward `4.47`.
  - cross geometry + normal phase: `0/10`, mean reward `14.28`.
  - normal geometry + null phase: `0/10`, mean reward `14.06`.

Rule:
- It is valid to present this as a learned positive diagnostic: true pointcloud geometry can drive a learned action decoder and beat no-depth/cross controls.
- The disentanglement control supports a stronger interpretation: both true object geometry and temporal phase are required; phase alone does not explain success.
- Do not present it as final OpenVLA RGB-D > RGB-only, because phase is supplied by a hand-written state machine and the representation uses segmentation-derived object features.
- Next real method should replace phase one-hot with learned temporal state via ACT, DP3/diffusion, recurrent policy, or DAgger/distillation.

Status:
Accepted

## D-013: Full projected heatmap residual is still too optional

Date: 2026-07-04 UTC

Reason:
- The formal `open_drawer` projected-heatmap policy run trained stably for `5000` steps.
- Paired normal-vs-cross-sample action diagnostics remained exactly zero across all action components.
- This rejects the idea that simply feeding full spatial logits into a bounded residual is enough to create causal depth usage.

Evidence:
- Full projected-heatmap paired diagnostic:
  - `paired_pred_l1=0.0`
  - `paired_pred_rmse=0.0`
  - `paired_pred_xyz_l2=0.0`
  - `paired_pred_rpy_l2=0.0`
  - `paired_pred_gripper_abs=0.0`

Rule:
- Do not continue scalar, UV, or 2D-heatmap residual sweeps as the main path.
- The next policy bottleneck must make 3D location selection structurally determine the translation action.
- Preferred next implementation: primary 3D waypoint/action-map head, with normal/null/cross-sample paired action-delta gating before rollout.

Status:
Accepted

## D-014: Bounded point-action residual is still too optional

Date: 2026-07-04 UTC

Reason:
- The formal `open_drawer` point-action gate trained stably for `5000` steps.
- It used dense point tokens, action/language-conditioned point scoring, `point_keypose_xyz` auxiliary supervision, and a bounded point-selected translation residual.
- Paired normal-vs-cross-sample action diagnostics remained exactly zero across all action components.

Evidence:
- Point-action paired diagnostic:
  - `paired_pred_l1=0.0`
  - `paired_pred_rmse=0.0`
  - `paired_pred_xyz_l2=0.0`
  - `paired_pred_rpy_l2=0.0`
  - `paired_pred_gripper_abs=0.0`
- Single-step action metrics were not enough to prove depth usage: `rmse=0.022144`, `xyz_rmse=0.001285`.

Rule:
- Do not continue bounded point-action residual sweeps as the main path.
- The next policy bottleneck must decode translation from a 3D action map, voxel/point action classification, waypoint prediction, or diffusion-style action head where corrupted depth changes the selected action.
- Continue rejecting any checkpoint whose paired normal-vs-cross-sample action delta is exactly zero before rollout.

Status:
Accepted

## D-002: Use RLBench as the first main benchmark

Date: 2026-07-03

Reason:
- RLBench official materials position it as a vision-guided manipulation benchmark for imitation learning, multi-task learning, and geometric computer vision.
- It provides RGB-D, camera geometry, language/task descriptions, gripper pose, and closed-loop evaluation.
- PerAct/RVT/Act3D provide strong 3D action-grounding references on RLBench-like settings.

Evidence:
- Official/project sources recorded in `RGBD_DATASET_ACTIONSPACE_RESEARCH.md`.
- Local RLBench environment, conversion, validation, probe, and eval entrypoints run.

Status:
Accepted

## D-003: Validate `reach_target` before scaling stable6 or 18-task training

Date: 2026-07-03

Reason:
- Demo replay succeeds with enough horizon, so action mode is not fundamentally broken.
- Current stable6 policy rollout fails through drift/planner infeasibility.
- A single-task overfit is the smallest useful gate before spending compute on scaling.

Evidence:
- Demo replay success: `6/6` with max_steps 200.
- Matched stable6 policy eval: RGB-only and RGB-D normal both `0/6`.
- `reach_target` subset validated at `/root/RLBench/rgbd_hdf5_reach_3demos_64`.

Status:
Accepted

## D-004: Keep delta execution but add absolute keypose/depth grounding

Date: 2026-07-03

Reason:
- Delta action chunks are compatible with OpenVLA-OFT and usually easier to learn.
- Depth's advantage is metric 3D geometry; absolute keypose or action maps make depth useful to the loss.
- CALVIN confirms absolute/relative action spaces are worth comparing.

Evidence:
- RLBench converter stores `rlbench_keypose_action`.
- Offline keypose probe shows normal depth beats null/shuffle on absolute keypose prediction.

Status:
Accepted

## D-005: Treat saturated benchmarks as automatic non-claim benchmarks

Date: 2026-07-03

Reason:
- A benchmark where matched RGB-only is already near ceiling cannot reveal the marginal value of metric depth.
- Optimizing depth fusion on such a benchmark encourages chasing aggregate success noise instead of causal 3D usage.
- The project needs a hard rule to avoid drifting back to LIBERO because it is familiar and cheap.

Rule:
- If RGB-only is near saturated, the benchmark can be used for smoke, regression, or historical comparison only.
- Main claims require a non-saturated closed-loop benchmark plus matched RGB-only and normal/null/shuffle depth controls.

Status:
Accepted

## D-006: Use cross-sample depth corruption for dense point causal tests

Date: 2026-07-03

Reason:
- Pixel shuffle preserves too much point-set/statistical structure for dense point tokens.
- Reach-only RGB-D diagnostic showed normal and pixel shuffle were nearly identical.
- A stronger corruption should replace depth/camera geometry with another sample or episode.

Rule:
- Keep pixel shuffle as a weak diagnostic only.
- Use `cross_sample` / `shuffle_samples` for dense-point causal gates before scaling.

Status:
Accepted

## D-007: Repair RGB-D through protected residual action fusion before scaling

Date: 2026-07-03

Reason:
- Web search across PointVLA, BridgeVLA, Act3D, PerAct/RVT, and 3D Diffusion Policy consistently points to observation-action spatial alignment rather than naive feature append.
- Local evidence shows RGB-only `reach_target` succeeds, while RGB-D normal fails and RGB-D null succeeds. The next problem is not benchmark selection alone; it is harmful depth-action fusion.
- PointVLA supports freezing/protecting the vanilla action expert and injecting 3D through lightweight modules.
- BridgeVLA and Act3D support explicit heatmap/keypose/3D action-map alignment.

Rule:
- Do not scale stable6/18-task RGB-D until reach-only RGB-D normal passes the causal gate.
- The default next implementation should be a bounded, low-gate depth residual over a working RGB action path, with frozen RGB anchor modules first.
- Normal depth must beat or at least match null on `reach_target` before any larger training run.

Status:
Accepted

## D-008: Treat reach-only RGB-D success as sanity, not depth evidence

Date: 2026-07-04 UTC

Reason:
- Safe RGB-anchor RGB-D reach succeeds with normal depth, but null and cross-sample depth also succeed with identical rollout statistics.
- Offline diagnostics are identical for normal/null/cross-sample depth, so the model is not causally using depth content.
- `reach_target` is too easy for the current RGB/proprio action expert after overfitting.

Rule:
- Keep `reach_target` for regression: RGB-only and RGB-D should not break.
- Do not claim depth gain from reach-only success.
- Next depth-gain evidence must come from a 3D-sensitive task or action target where corrupted depth fails.

Status:
Accepted

## D-009: Shallow safe residual is a no-go for depth gain

Date: 2026-07-04 UTC

Reason:
- On `open_drawer`, RGB-only does not saturate and normal depth has strong offline keypose signal.
- Safe RGB-D still matches RGB-only rollout failure: all modes `0/1` at horizon `200`.
- Normal and cross-sample depth produce identical action diagnostics, so the model is not using real depth geometry.

Rule:
- Keep RGB-anchor protection as a safety mechanism, not as the main depth-learning mechanism.
- Do not continue simple low-gate/clamp/residual sweeps as the primary path.
- Next implementation must make spatial prediction part of action formation: keypose-conditioned residual, heatmap, or 3D action map.

Status:
Accepted

## D-010: Use projected keypose heatmap as the next depth-action bottleneck

Date: 2026-07-04 UTC

Reason:
- Keypose-conditioned scalar residual trained stably but still produced identical normal and cross-sample actions.
- Projected keypose heatmaps create a spatial output where corrupted depth is measurably worse before policy rollout.
- This matches the BridgeVLA/Act3D lesson: depth should solve an action-grounded spatial target, not merely provide optional hidden features.

Evidence:
- `open_drawer` projected-keypose heatmap probe:
  - normal peak error `2.85px`
  - cross-sample peak error `8.86px`
  - null peak error `43.30px`
  - paired normal-vs-cross peak delta `8.11px`
- stable6 projected-keypose heatmap probe:
  - normal peak error `2.94px`
  - cross-sample peak error `12.53px`
  - null peak error `57.63px`
  - paired normal-vs-cross peak delta `12.13px`

Rule:
- Next training path should wire projected heatmap output into final action formation.
- Do not treat heatmap auxiliary loss alone as sufficient; it must change the predicted action under cross-sample depth.
- Continue using matched RGB-only, normal, null, and cross-sample depth gates.

Status:
Accepted

## D-011: Projected UV coordinates alone are not enough

Date: 2026-07-04 UTC

Reason:
- The projected-keypose heatmap probe separates normal depth from null/cross-sample depth, so the spatial target is meaningful.
- A full `open_drawer` projected-UV residual policy run trained stably for `5000` steps, but paired normal-vs-cross-sample action predictions were exactly identical.
- Compressing the spatial evidence to four normalized UV scalars leaves the final action residual optional and easy to ignore.

Evidence:
- Projected-UV paired diagnostic:
  - `paired_pred_l1=0.0`
  - `paired_pred_rmse=0.0`
  - `paired_pred_xyz_l2=0.0`
  - `paired_pred_rpy_l2=0.0`
  - `paired_pred_gripper_abs=0.0`

Rule:
- Do not scale projected-UV coordinate residuals.
- The next implementation must pass spatial distribution or 3D action-map features into final action formation, not only coordinate labels.
- Keep the paired normal-vs-cross action-delta gate as mandatory before rollout.

Status:
Accepted

## D-012: Use full projected heatmap logits for the next policy gate

Date: 2026-07-04 UTC

Reason:
- Projected heatmap probes showed real normal-vs-corrupt depth separation, while four-value projected UV residuals were ignored by the final action path.
- Full heatmap logits preserve spatial uncertainty and distributional structure that is lost in a coordinate bottleneck.
- Feeding both heatmap logits and soft-argmax UV coordinates into the bounded action residual is the smallest next step before moving to a full 3D action map.

Evidence:
- Implemented `aux_target=projected_keypose_heatmap` with two-view `16x16` Gaussian labels.
- Action residual receives `512` heatmap logits plus `4` soft-argmax coordinates.
- Verification passed: synthetic dataset smoke, action-head tensor smoke, and real `open_drawer` `MAX_STEPS=1` training smoke with aux prediction/label shape `(1, 2, 16, 16)`.

Rule:
- The next formal `open_drawer` run should use `DEPTH_AUX_TARGET=projected_keypose_heatmap`, `DEPTH_AUX_OUTPUT_DIM=512`, `DEPTH_AUX_HEATMAP_SIZE=16`, and `DEPTH_AUX_HEATMAP_SIGMA=1.5`.
- Run paired normal-vs-cross action diagnostic before rollout.
- If paired action deltas remain zero, reject this path and move to a coarse 3D action map.

Status:
Accepted
