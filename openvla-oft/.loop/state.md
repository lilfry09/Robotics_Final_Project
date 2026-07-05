# Loop State

## Current Goal

Continue the OpenVLA end-to-end RGB-D route by making depth enter a diagnosable object/contact geometry bottleneck before it affects action.

## Current Focus

- Main diagnostic benchmark: RLBench. LIBERO is no longer a main dataset; it is only a sanity/regression benchmark. The next positive-result route has moved to ManiSkill3 data adapter plus a real point-cloud action decoder.
- Current minimum experiment result: safe reach-only RGB-D with RGB-anchor protection completed. It restores reach rollout success, but fails the depth causal gate because normal/null/cross-sample are identical.
- Latest `open_drawer` action-grounding results: keypose-conditioned residual, projected-UV residual, full projected-heatmap residual, point-action attention, and primary waypoint-action all trained stably to `5000` steps, but none produced a defensible RGB-D depth gain.
- Latest projected heatmap auxiliary result: RLBench projected-keypose heatmap probe passed on both `open_drawer` and stable6. Stable6 normal peak error `2.94px`, cross-sample `12.53px`, null `57.63px`.
- Latest full heatmap policy gate result: no-go. A full `open_drawer` 5000-step projected-heatmap residual run trained stably, but paired normal-vs-cross-sample action delta was exactly `0.0`.
- Last tested policy representation: dense point depth tokens + action/language-conditioned point selection + `farthest_future_pose_xyz` auxiliary target + primary waypoint xyz action override.
- Current risk: even when selected 3D point directly controls first-step xyz, the learned action can be only weakly depth-sensitive and still fail closed-loop through planner-infeasible actions.
- Current implementation decision: stop this recipe. Treat the final farthest-future 5000-step waypoint run as a no-go: it created only a tiny normal-vs-cross action delta, and strict normal/cross diagnostics remained effectively tied.
- New action-map feasibility result: short-horizon keypose/next-pose labels have an EE-position shortcut, but future/final/farthest-future targets make normal point candidates beat both cross-sample candidates and EE fallback.
- Final pre-submission code update: `future_pose_xyz`, `final_pose_xyz`, and `farthest_future_pose_xyz` are wired into training, config saving, and the RLBench stage runner. These are verified by label smoke, runner dry-run, and a `MAX_STEPS=1` real training smoke; no positive RGB-D result is claimed from them.
- Latest long-horizon target test: a 5000-step `farthest_future_pose_xyz` primary waypoint run completed, but failed the causal gate. Paired normal-vs-cross `paired_pred_xyz_l2=1.00e-04`; strict normal/cross `xyz_rmse` was effectively tied, with cross-sample slightly better (`0.003167` vs `0.003162`).
- Current decision: do not rollout or scale the current farthest-future waypoint recipe. It did not produce causal normal-depth action dependence after 500 or 5000 steps.
- New method-search conclusion: dataset change is necessary but not sufficient; after RLBench residual/waypoint no-gos, the next route should combine ManiSkill3-style higher-throughput RGB-D/pointcloud data with a primary point-cloud action decoder, not another optional OpenVLA-OFT residual.
- Latest ManiSkill3 pilot: official `PushCube-v1` and `PickCube-v1` demos were replayed into pointcloud + `pd_ee_delta_pos` action. A tiny PointNet-style decoder completed offline normal/null/cross gates. `PushCube-v1` passed strict pointcloud gate in `3/3` seeds with mean paired normal-vs-cross L2 `0.002041`; `PickCube-v1` passed in `2/3` seeds with mean paired normal-vs-cross L2 `0.022263`. A 5000-step PickCube model increased paired action sensitivity to `0.0373`, but closed-loop smoke still failed: normal/null/cross all `0/3`. A minimal 8-step action chunk decoder also passed offline gate but failed closed-loop (`0/3` normal, cross_demo reward tied). A goal-conditioned PointNet still failed the strict offline gate and closed-loop. An object-centric feature MLP strongly passed offline causal gate (`paired_normal_vs_cross_l2=1.314387`) but still failed closed-loop (`0/10`). A pointcloud geometry controller is a positive diagnostic: normal pointcloud solved PickCube `7/10` at 100 steps and `8/10` at 150 steps, while null/cross controls were far behind (`0/10` and `1/10` at 150 steps).
- Latest learned positive diagnostic: phase-conditioned geometry-teacher distillation first solved normal `17/30`, null `0/30`, cross_demo `0/30`. A follow-up single-frame phase classifier reached validation accuracy `96.1%` and replaced the hand-written phase during rollout; learned-phase normal solved `19/30`, while null and cross_demo stayed `0/30`. This is still not an OpenVLA/raw-RGB-D result because the policy uses segmentation-derived object features, but it proves learned actions can exploit true geometry with learned phase prediction.
- Latest stricter control: learned-phase disentanglement shows null geometry + learned normal phase, cross geometry + learned normal phase, and normal geometry + learned null phase are all `0/10`. This rejects the easy explanation that phase alone caused success.
- Raw pointcloud follow-up, 30 teacher episodes: replacing `cube_center` input with `z>0.02` cropped pointcloud + task state, plus phase and cube-center auxiliary heads, passes the offline causal gate strongly. Normal cube RMSE is `0.009m`, cross/null are about `0.075m`; paired normal-vs-cross action L2 is `0.215`. Closed-loop is weak but positive: normal `2/30`, null `0/30`, cross_demo `0/30`.
- Latest scaled raw pointcloud result: expanding geometry-teacher data to `100` successful episodes (`8388` transitions) and training an h256/10k single-step raw-pointcloud policy gives a clear learned closed-loop gain. Offline normal action RMSE is `0.067`, null `0.210`, cross `0.131`; paired normal-vs-cross action L2 is `0.120`. Across two 30-episode eval seeds, normal solves `20/60`, while eval-time null and cross_demo are `1/60` each. Matched train-time no-depth baselines are also far lower: sampled-RGB-only `1/60` and null/proprio `3/60`. This is the strongest raw-pointcloud learned action result so far, but still a ManiSkill teacher-distilled policy rather than OpenVLA.
- Latest action-chunk control: the success100 h=8 action-chunk model passes the offline gate (`paired_normal_vs_cross_step_l2=0.188`) but does not improve closed-loop success over the single-step decoder. This suggests simple chunking is not enough; future temporal policy should be ACT/DP3/diffusion/recurrent rather than the shallow chunk head.
- Latest perception/action bottleneck split: using the same learned raw-pointcloud cube predictor inside the fixed geometry controller solves normal `22/30`, while null is `1/30` and cross_demo `0/30`. This proves raw pointcloud perception is strong enough for control; the remaining gap is learned action/temporal decoding quality, not geometry availability.
- New OpenVLA end-to-end continuation: added `visible_object_point_xyz` and `visible_object_rel_xyz` auxiliary targets. `visible_object_point_xyz` supervises dense point selection from current RGB-D and can feed the waypoint primary xyz action. The lightweight smoke `experiments/robot/rlbench/smoke_depth_geometry_bottleneck.py` passes and verifies that selected geometry is written into first-step action.
- Latest OpenVLA geometry-bottleneck result: a scale-aware `open_drawer` 500-step run with `visible_object_point_xyz`, `DEPTH_WAYPOINT_ACTION_CLIP=0.02`, and `DEPTH_WAYPOINT_ACTION_SCALE=20.0` passed the offline causal action gate. Paired normal-vs-cross: `paired_depth_point_xyz_l2=0.1988`, `paired_depth_waypoint_xyz_action_l2=0.6776`, `paired_pred_xyz_l2=0.00743m`. Strict diagnostic: normal `xyz_rmse=0.00502`, cosine `0.3608`; null `xyz_rmse=0.00603`, cosine `0.1140`; cross-sample `xyz_rmse=0.00556`, cosine `0.2356`. This is not rollout superiority yet, but it proves depth now enters OpenVLA final action with a measurable normal-depth advantage over null/cross controls.
- Latest OpenVLA closed-loop sanity: same scale20 checkpoint with normal depth on `open_drawer` got `0/1`, length `193`, `InvalidActionError`, mean xyz step `0.00865`. Added eval-only `MAX_DELTA_RPY`; with `MAX_DELTA_RPY=0.02`, normal ran full horizon (`0/1`, length `200`, no error), null failed quickly (`0/1`, length `4`, `InvalidActionError`), and cross_sample also ran full horizon (`0/1`, length `200`, no error). Eval `wpscale=30` normal also ran full horizon but still `0/1`. Keep the claim at offline causal action gate plus closed-loop stability separation; do not claim rollout success.
- Latest OpenVLA gripper diagnostic: open_drawer demos close the gripper around step `73-78`. The learned normal rollout has mean gripper command `0.888` (too open), but an eval-only close-after-step-75 override lowers mean gripper command to `0.375` and still gives `0/1`, length `200`, no error. Gripper timing is a bottleneck but not sufficient; remaining issue is temporal/action decoder and object/contact-conditioned waypoint sequence.
- Latest OpenVLA unfrozen action-head follow-up: a 1000-step run with `FREEZE_ACTION_HEAD_BASE=False` improved offline action imitation and depth separation. Paired normal-vs-cross: `paired_depth_point_xyz_l2=0.2011`, `paired_depth_waypoint_xyz_action_l2=0.6293`, `paired_pred_xyz_l2=0.00690m`. Strict diagnostic: normal `xyz_rmse=0.00377`, cosine `0.6692`; null `xyz_rmse=0.00603`, cosine `0.1140`; cross_sample `xyz_rmse=0.00541`, cosine `0.2563`. Closed-loop did not improve: normal with `MAX_DELTA_RPY=0.02` failed at length `67`; `MAX_DELTA_RPY=0.005` failed at length `82`; forced-open gripper also failed at length `82`. Interpretation: depth-action coupling is real, but full RLBench task success is blocked by temporal/contact/action-decoder stability, not by depth being absent from final action.
- Latest OpenVLA temporal geometry follow-up: added `DEPTH_WAYPOINT_ACTION_CHUNK_LEN` so the depth-selected waypoint can overwrite xyz for multiple action chunk steps. Default remains `1`; chunk8 writes xyz into all 8 OpenVLA chunk steps. A 500-step frozen-base chunk8 run from the scale20 checkpoint passed diagnostics. Paired normal-vs-cross: `paired_depth_point_xyz_l2=0.1951`, `paired_depth_waypoint_xyz_action_l2=0.6024`, `paired_depth_waypoint_chunk_xyz_action_l2=1.7038`, `paired_pred_xyz_l2=0.00678m`. Strict diagnostic: normal `xyz_rmse=0.00414`, cosine `0.6125`; null `xyz_rmse=0.00603`, cosine `0.1140`; cross_sample `xyz_rmse=0.00541`, cosine `0.2584`. Closed-loop with default chunk execution (`action_chunk[0]` only) and `MAX_DELTA_RPY=0.02`: normal `0/1` length `200` no error, null length `4` `InvalidActionError`, cross_sample length `190` `InvalidActionError`. Added `ACTION_CHUNK_EXEC_HORIZON`; with true 8-step chunk execution, normal remains stable (`0/1`, length `200`, no error) while null still fails at length `4`. This upgrades the OpenVLA claim to depth entering temporal action geometry and affecting closed-loop stability, but still not task success or RGB-D superiority.
- Latest OpenVLA trace diagnostic: added `EVAL_TRACE_OUTPUT` JSONL logging for per-step EE xyz, target xyz, delta action, gripper command, selected depth point, and current waypoint chunk xyz. Short exec8 trace on the chunk8 checkpoint: normal produced 32 trace rows with chunk index `0..7` repeated and new predictions at `0,8,16,24`; null produced 4 rows then `InvalidActionError`. First-step normal-vs-null trace differences: selected depth point L2 `0.6205m`, waypoint chunk L2 `0.8547`, final xyz delta L2 `0.00655m`. Added `analyze_eval_trace.py`; it shows normal EE-to-depth-point distance drops from `0.2780m` to `0.0144m`, min `0.0097m`, with action-depth direction cosine mean `0.8103`, while null remains about `0.55m` from its selected point. Gripper never closes below `0.5` in either trace. This gives closed-loop evidence that the depth geometry signal is present at execution time and that the current remaining bottleneck is contact/gripper/trajectory completion, not missing depth-action coupling.
- Latest OpenVLA depth-near gripper diagnostic: added eval-only `GRIPPER_OVERRIDE_MODE=latch_close_near_depth_point`. With true chunk execution and close threshold `0.03m`, normal closes at step `27`, reaches min EE-to-depth-point `0.00464m`, and runs full horizon (`0/1`, length `200`, no error); null still fails at length `4` with `InvalidActionError`. This fixes gripper timing diagnostically but still does not solve `open_drawer`, so the remaining bottleneck is not simply "the gripper never closed".
- Final OpenVLA contact-target attempt: added eval-only post-close pull controls (`POST_CLOSE_PULL_DELTA_XYZ`, `POST_CLOSE_PULL_STEPS`) and a new `first_close_pose_xyz` auxiliary target. Visible-object + close + pull `+Y` closed and pulled `10` steps but failed at length `38`; trace showed close height around `z=1.26`, above the demo contact height `z≈1.035`. A 500-step `first_close_pose_xyz` run trained from the chunk8 checkpoint kept a causal action signal (`paired_pred_xyz_l2=0.00659m`, `paired_depth_point_xyz_l2=0.1594`, selected-point mean `z=1.102`) and first-step rollout selected a handle-height point (`z≈1.039`). However closed-loop point selection drifted by `0.3408m` and later selected points moved toward wrong low regions (`z≈0.83-0.88`). `MAX_DELTA_RPY=0` improved length from `15` to `61`; a wide close threshold `0.2m` closed at step `59` and pulled one step, but still failed with `InvalidActionError`. Final OpenVLA boundary: depth-action coupling is real, but stable object/contact grounding and post-contact temporal action are still missing.
- Latest OpenVLA latch/oracle diagnostic: added eval-only `DEPTH_POINT_LATCH_MODE=first|demo_first_close` plus `LATCHED_DEPTH_POINT_ACTION_STEP`. With learned first selected point latched, selected-point drift becomes `0`, EE-to-point distance drops from `0.473m` to min `0.00062m`, gripper closes at step `45`, and post-close `+Y` pull runs `35` steps; rollout runs full horizon but still `0/1`. This proves the point-drift instability can be removed, but the selected point is not a sufficient handle/contact pose. With oracle demo first-close point, the controller also runs full horizon but does not solve: close `0.03m` reaches only min `0.039m`, and close `0.05m` also stays `0/1`. The remaining bottleneck is now broader than point selection: executable gripper pose/orientation, contact constraints, and post-contact trajectory are missing.
- Latest OpenVLA demo-tail upper-bound diagnostic: added `eval_demo_replay.py` plus eval-only `POST_CLOSE_DEMO_TAIL_MODE=first_close`. Direct replay of stored `open_drawer` demo EE pose/gripper sequences succeeds `3/3` with mean length `92.33`, so the RLBench action mode/planner can execute expert trajectories. Hybrid oracle with strict close thresholds `0.03/0.05/0.08m` does not trigger demo tail, showing the straight-line point controller cannot reliably enter contact; a wide `0.20m` threshold with demo tail starting `20` frames before first close succeeds `1/1` at length `61`, activating at step `25` from demo index `54`. This is not learned-policy success; it shows the opportunity window is stable pre-contact approach plus temporal/contact action decoding.
- Latest learned-first demo-tail control: replacing oracle demo first-close point with the policy's first selected depth point gives normal `1/1` at the wide `0.20m` gate, length `64`, while null fails at length `2` with `InvalidActionError`. However cross_sample and shuffle also solve (`1/1`, lengths `60/61`), and a tighter `0.195m` gate still gives normal/cross `1/1`. Therefore this is not a strict normal-vs-corrupt success; it means the selected-point bottleneck can provide a rough geometric hook for expert tail, but it is not yet object/contact-specific enough.
- New OpenVLA target wiring: added `pre_first_close_pose_xyz`, which labels the EE xyz `aux_future_horizon` frames before first gripper close. With `aux_future_horizon=20`, real `open_drawer` demo0 maps first close index `74` to pre-contact index `54`, xyz `[0.2118, 0.0601, 1.0512]`, matching the demo-tail successful entry. Synthetic dataset smoke, action-head smoke, and `MAX_STEPS=1` real training smoke pass. The one-step training printed aux target `pre_first_close_pose_xyz`, prediction/label shape `(1,3)`, finite aux loss `0.0459`; small paired diagnostic loaded the checkpoint and produced `paired_depth_point_xyz_l2=0.1888`, `paired_pred_xyz_l2=0.00819m`. The temporary 855M smoke checkpoint was deleted; only `experiments/logs/pre_first_close_smoke_diag/` remains.
- Latest OpenVLA pre-contact target gate: trained `pre_first_close_pose_xyz` for 500 steps from the `first_close_pose_xyz` chunk8 checkpoint, then deleted the no-go checkpoint to save about `855M` after preserving JSON diagnostics. Paired normal-vs-cross still shows action/geometry coupling (`paired_depth_point_xyz_l2=0.1686`, `paired_depth_waypoint_chunk_xyz_action_l2=1.6105`, `paired_pred_xyz_l2=0.00690m`). Strict diagnostics show normal beats null (`xyz_rmse=0.00382` vs `0.00491`, cosine `0.588` vs `0.251`), but cross_sample beats normal (`xyz_rmse=0.00357`, cosine `0.753`). Decision: do not claim OpenVLA learned RGB-D success and do not scale this recipe; the remaining issue is object/contact-specific grounding plus temporal/contact action decoding.
- Latest OpenVLA visible pre-contact geometry gate: added `visible_first_close_point_xyz` and `visible_pre_first_close_point_xyz`, plus aux-label error reporting in `diagnose_policy_actions.py`. The target is the current RGB-D visible point nearest to the demo pre-close EE xyz, so the label is tied to current depth geometry. Feasibility probe shows `visible_pre_first_close_point` normal median coverage `0.0000m`, cross `0.0142m`, EE fallback `0.0934m`; direct `pre_first_close_pose` did not have this property. A 500-step `visible_pre_first_close_point_xyz` run from the first-close chunk8 checkpoint completed and is kept at `runs_rlbench_visible_preclose_500/`. Selected-point geometry passes normal/null/cross: normal selected point -> aux label L2 `0.099m`, null `0.699m`, cross_sample `0.194m`; paired normal-vs-cross selected-point advantage `0.095m`, `paired_depth_point_xyz_l2=0.1746m`, `paired_depth_waypoint_chunk_xyz_action_l2=1.8790`, `paired_pred_xyz_l2=0.00818m`. However strict action imitation still has cross_sample better (`xyz_rmse=0.00364`, cosine `0.722`) than normal (`xyz_rmse=0.00395`, cosine `0.561`), so this is a positive OpenVLA geometry-bottleneck result, not rollout success or RGB-D superiority.

## Active Tasks

- [x] T-001: Finish reach-only RGB-only overfit training and evaluate closed-loop success.
- [x] T-002: Train reach-only RGB-D dense/keypose checkpoint after RGB baseline finishes.
- [x] T-003: Evaluate reach-only RGB-D with normal/null/shuffle depth and compare against RGB-only.
- [x] T-004: Repair harmful RGB-D fusion on reach-only so it no longer breaks RGB behavior.
- [x] T-005: Add cross-sample depth corruption for dense-point rollout/diagnostic.
- [x] T-006: Move from reach sanity to a 3D-sensitive task/action target where normal depth must beat null/cross-sample.
- [x] T-009: Train and diagnose `open_drawer` keypose-conditioned residual before any larger RLBench/ManiSkill scaling.
- [x] T-010: Prototype projected heatmap grounding on `open_drawer`/stable6; require normal depth to differ from and beat null/cross-sample before rollout.
- [x] T-011: Train projected-UV residual `open_drawer` policy gate and require nonzero paired normal-vs-cross action delta before rollout.
- [x] T-012: Wire full projected heatmap logits/soft-argmax into action formation and require nonzero paired normal-vs-cross action delta.
- [x] T-013: Train the `projected_keypose_heatmap` `open_drawer` policy gate and reject it unless paired normal-vs-cross action delta is nonzero.
- [x] T-014: Prototype a point-action attention gate where selected 3D location directly determines a bounded translation residual.
- [x] T-015: Prototype a coarse 3D action map or waypoint action head where the selected 3D target is the primary translation action, not an optional residual.
- [x] T-016: Wire long-horizon RLBench xyz auxiliary targets after feasibility probe showed short-horizon EE/proprio shortcut.
- [x] T-017: Run a small `farthest_future_pose_xyz` waypoint gate and reject it unless normal beats null/cross-sample.
- [x] T-018: Run a 5000-step `farthest_future_pose_xyz` confirmation gate and reject it unless normal beats null/cross-sample.
- [x] T-019: Build a ManiSkill3 adapter smoke and export one RGB-D/pointcloud task to unified HDF5.
- [x] T-020: Train a lightweight point-cloud action decoder pilot and require normal > null/cross before rollout.
- [x] T-021: Turn the ManiSkill3 point-cloud decoder pilot into a closed-loop diagnostic with matched null/cross controls.
- [x] T-022: Replace hand-written PickCube phase with a learned phase classifier and re-run normal/null/cross plus disentanglement controls.
- [x] T-023: Move the PickCube teacher distillation from segmentation-derived object features toward raw cropped pointcloud with cube-center auxiliary supervision.
- [x] T-024: Split raw pointcloud perception from action decoding by feeding learned cube predictions into the fixed geometry controller.
- [x] T-025: Scale raw cropped-pointcloud teacher distillation from 30 to 100 successful episodes and test single-step vs action-chunk learned decoders.
- [x] T-026: Add an OpenVLA visible-object geometry bottleneck target and smoke-test that selected depth geometry can become primary xyz action.
- [x] T-027: Train `open_drawer` OpenVLA with `visible_object_point_xyz` and require paired normal-vs-cross geometry/action deltas before rollout.
- [x] T-028: Run unfrozen action-head OpenVLA follow-up and diagnose offline action coupling versus closed-loop stability.
- [x] T-029: Add 8-step waypoint chunk geometry path and verify normal/null/cross offline plus closed-loop stability separation.
- [x] T-030: Add per-step RLBench eval trace logging and verify depth-selected geometry appears during true chunk execution.
- [x] T-031: Add eval-only depth-near gripper and post-close pull diagnostics to separate gripper timing from contact/pull trajectory.
- [x] T-032: Add and test `first_close_pose_xyz` contact supervision; reject it for task success but keep the narrowed failure diagnosis.
- [x] T-033: Add eval-only latched selected-point and oracle demo first-close diagnostics; verify fixed xyz contact alone is insufficient for task success.
- [x] T-034: Add demo replay and demo-tail oracle upper-bound diagnostics; verify the eval stack can solve stored expert trajectories and that expert temporal tail can complete once a wide pre-contact gate is reached.
- [x] T-035: Run learned-first selected-point demo-tail normal/null/cross/shuffle controls; reject it as strict depth causal success because cross/shuffle also complete under the wide gate.
- [x] T-036: Add `pre_first_close_pose_xyz` pre-contact auxiliary target and verify dataset, action-head, one-step training, and tiny diagnostic paths.
- [x] T-037: Train 500-step `pre_first_close_pose_xyz` gate and reject it as strict OpenVLA success because cross-sample still beats normal.
- [x] T-038: Add `visible_pre_first_close_point_xyz`, train 500 steps, and verify selected-point geometry beats null/cross while action imitation still fails strict cross gate.

## Done

- [x] Datasets researched from official/project sources: RLBench, ManiSkill3, CALVIN, RoboCasa365, PerAct, RVT, Act3D.
- [x] Methods refreshed from official/project/arXiv sources: 3D Diffusion Policy, PointVLA, SpatialVLA, BridgeVLA.
- [x] LIBERO demoted to sanity-check/history role because RGB-only is saturated.
- [x] Saturated benchmark rule accepted: no main depth-gain claim from near-ceiling RGB-only benchmarks.
- [x] RLBench environment, data conversion, validation, offline keypose probe, and rollout entrypoints are working.
- [x] `stable6 x 3 demos` generated and converted: `18 demos / 2009 transitions`.
- [x] `reach_target` HDF5 subset created: `/root/RLBench/rgbd_hdf5_reach_3demos_64`.
- [x] Offline policy-vs-demo diagnostic added and run for RGB-only and RGB-D normal/null/shuffle.
- [x] Reach-only RGB baseline overfit passed: `1/1` success for `MAX_DELTA_XYZ=0.03/0.05/0.08`, length `29`.
- [x] RLBench stage runner now exposes `DEPTH_AUX_SPATIAL_LOSS_WEIGHT` for reproducible depth-coupling sweeps.
- [x] Reach-only RGB-D aux `0.2` trained to `5000` steps; rollout `NO-GO`: normal `0/1`, null `1/1`, shuffle `0/1` at `MAX_DELTA_XYZ=0.05`.
- [x] Added `cross_sample` depth corruption to offline diagnostic and RLBench rollout eval.
- [x] Added safe RGB-D runner controls for gate init, depth dropout, and freezing RGB/proprio/action-head base modules.
- [x] Added optional depth hidden/action residual clamps and wired them through train/eval/diagnostic runners.
- [x] Safe RGB-anchor RGB-D reach run completed with `RESUME_COMPONENTS_FROM` set to the successful RGB-only checkpoint.
- [x] Safe RGB-anchor reach rollout: normal/null/cross-sample all `1/1`, length `29`, identical action norms.
- [x] Safe RGB-anchor offline diagnostic: normal/null/cross-sample all `xyz_rmse=0.001700`, `xyz_direction_cosine=0.97756`.
- [x] Deleted obsolete failed `/root/runs_rlbench_stable6_3demos` checkpoint directory to free about `1.5G`.
- [x] Deleted obsolete LIBERO-era DepthVLA run directories in `openvla-oft`, freeing `autodl-tmp` from `4.7G` to `13G` available.
- [x] Built `open_drawer` HDF5 subset at `/root/RLBench/rgbd_hdf5_open_drawer_3demos_64` with `3 demos / 317 transitions`.
- [x] `open_drawer` dense-depth keypose probe passed: normal xyz RMSE `0.0705`, null `0.1420`, shuffle `0.1466`.
- [x] `open_drawer` RGB-only single-task baseline trained to `5000` steps in `/root/runs_rlbench_open_drawer_3demos`.
- [x] `open_drawer` RGB-only eval at horizon `200`, `MAX_DELTA_XYZ=0.05`: `0/1`, length `200`, no `InvalidActionError`.
- [x] Trained `open_drawer` safe RGB-D from the RGB-only anchor and evaluated normal/null/cross-sample.
- [x] `open_drawer` safe RGB-D rollout: normal/null/cross-sample all `0/1`, length `200`, no `InvalidActionError`.
- [x] `open_drawer` safe RGB-D diagnostic: normal and cross-sample identical (`xyz_rmse=0.001336`, cosine `0.78733`); null only slightly worse.
- [x] Deleted no-go safe RGB-D checkpoints for reach/open_drawer after recording results; kept only useful RGB-only anchors.
- [x] Implemented keypose-conditioned residual knobs: `depth_keypose_residual_weight` and `depth_keypose_residual_clip`, wired through train/eval/diagnose/RLBench runner.
- [x] Added paired diagnostic option `DIAG_COMPARE_DEPTH_MODE=cross_sample` to measure normal-vs-corrupt action deltas on the same samples before rollout.
- [x] Verification passed: py_compile, runner bash syntax, runner dry-run with keypose residual args, tensor shape smoke for `(B,8,7)` action residual, and real `MAX_STEPS=1` open_drawer training smoke.
- [x] Cleaned disk: purged pip cache and deleted the obsolete reach-only RGB-D no-go checkpoint. `/root` free space increased from about `6.1G` to about `12G`.
- [x] Trained `open_drawer` keypose-conditioned residual to `5000` steps in `/root/runs_rlbench_open_drawer_keypose_residual`.
- [x] Paired diagnostic for keypose residual failed causal gate: normal vs cross-sample `paired_pred_l1=0.0`, `paired_pred_rmse=0.0`, `paired_pred_xyz_l2=0.0`.
- [x] Strict diagnostic confirmed no useful depth separation: normal RMSE `0.0137849`, null RMSE `0.0137148`, cross-sample RMSE `0.0137849`.
- [x] Cleaned old LIBERO data/assets after demoting LIBERO from the main benchmark: removed `LIBERO/libero/datasets`, `LIBERO-plus/libero/libero/assets`, and `LIBERO-plus-downloads`. `/root/autodl-tmp` free space increased from about `13G` to about `33G`.
- [x] Deleted the no-go `open_drawer` keypose-residual checkpoint after recording results; kept the useful reach/open-drawer RGB-only anchors.
- [x] Added `experiments/robot/rlbench/probe_projected_keypose_heatmap.py` and runner command `projected-heatmap-probe`.
- [x] `open_drawer` projected-keypose heatmap probe passed: normal peak error `2.85px`, cross-sample `8.86px`, null `43.30px`; paired normal-vs-cross peak delta `8.11px`.
- [x] stable6 projected-keypose heatmap probe passed: normal peak error `2.94px`, cross-sample `12.53px`, null `57.63px`; paired normal-vs-cross peak delta `12.13px`.
- [x] Added trainable `projected_keypose_uv` auxiliary target: projects `rlbench_keypose_action[:3]` into both RLBench camera views as normalized UV `[agent_u, agent_v, wrist_u, wrist_v]`.
- [x] Runner now exposes `DEPTH_AUX_TARGET` and `DEPTH_AUX_OUTPUT_DIM`; `projected_keypose_uv` uses `DEPTH_AUX_OUTPUT_DIM=4`.
- [x] Synthetic HDF5 smoke passed for `projected_keypose_uv`, and real `open_drawer` `MAX_STEPS=1` training smoke passed with aux prediction/label shape `(1, 4)`.
- [x] Trained `open_drawer` projected-UV residual to `5000` steps from the RGB-only anchor in `/root/runs_rlbench_open_drawer_projected_uv`.
- [x] Paired diagnostic for projected-UV residual failed causal gate: normal vs cross-sample depth produced `paired_pred_l1=0.0`, `paired_pred_rmse=0.0`, `paired_pred_xyz_l2=0.0`.
- [x] Implemented `projected_keypose_heatmap` auxiliary target with two-view Gaussian heatmap labels generated from `rlbench_keypose_action` and camera intrinsics/extrinsics.
- [x] Action residual now receives full projected heatmap logits plus soft-argmax UV coordinates when `aux_output_dim=2*size*size`.
- [x] Verification passed for projected heatmap policy path: py_compile, runner bash syntax, synthetic dataset smoke, action-head tensor smoke, and real `open_drawer` `MAX_STEPS=1` training smoke with aux prediction/label shape `(1, 2, 16, 16)`.
- [x] Trained `open_drawer` full projected-heatmap residual to `5000` steps from the RGB-only anchor in `/root/runs_rlbench_open_drawer_heatmap`.
- [x] Paired diagnostic for full projected-heatmap residual failed causal gate: normal vs cross-sample depth produced `paired_pred_l1=0.0`, `paired_pred_rmse=0.0`, `paired_pred_xyz_l2=0.0`.
- [x] Implemented point-action attention over dense point tokens with `point_keypose_xyz` auxiliary supervision and a bounded point-selected translation residual.
- [x] Fixed RLBench negative focal-length backprojection bug; dense point coordinates are meter-scale and real `open_drawer` one-step point aux loss dropped to about `0.0327`.
- [x] Trained `open_drawer` point-action gate to `5000` steps in `/root/runs_rlbench_open_drawer_point_action`.
- [x] Paired diagnostic for point-action gate failed causal gate: normal vs cross-sample depth produced `paired_pred_l1=0.0`, `paired_pred_rmse=0.0`, `paired_pred_xyz_l2=0.0`.
- [x] Implemented primary waypoint xyz action override with `depth_waypoint_action_weight` and `depth_waypoint_action_clip`, wired through train/eval/diagnostic runners.
- [x] Trained `open_drawer` waypoint-action gate to `5000` steps in `/root/runs_rlbench_open_drawer_waypoint_action`.
- [x] Paired diagnostic for waypoint-action gate produced a tiny nonzero depth effect: `paired_pred_l1=5.90e-05`, `paired_pred_rmse=1.15e-04`, `paired_pred_xyz_l2=3.05e-04`.
- [x] Strict diagnostic remained effectively tied: normal `xyz_rmse=0.003178`, null `0.003210`, cross-sample `0.003226`.
- [x] Closed-loop waypoint-action rollout failed for all depth modes: normal `0/1` length `11`, null `0/1` length `10`, cross-sample `0/1` length `11`, all with `InvalidActionError`.
- [x] Deleted the no-go waypoint-action checkpoint after recording JSON results; `/root` has about `11G` free and `/root/autodl-tmp` has about `33G` free.
- [x] Added `scripts/collect_depthvla_final_results.py`, `FINAL_RESULTS_TABLE.md`, and `EXPERIMENTS.md` so final evidence and reproduction commands are one-command collectable.
- [x] Added `probe_3d_action_map_feasibility.py` and ran it on `open_drawer`/stable6; it shows 3D point candidates have signal versus cross-sample, but current labels are too close to EE.
- [x] Added `future_pose_xyz`, `final_pose_xyz`, and `farthest_future_pose_xyz` training targets plus `DEPTH_AUX_FUTURE_HORIZON` runner support; `farthest_future_pose_xyz` passed a one-step training smoke with `(1, 3)` prediction/label shapes.
- [x] Ran a 500-step `farthest_future_pose_xyz` gate. It was no-go: paired action delta remained tiny and strict normal/cross diagnostics were tied.
- [x] Ran a 5000-step `farthest_future_pose_xyz` confirmation gate. It was no-go: paired action delta shrank to `1.00e-04`, and strict cross-sample `xyz_rmse=0.003162` was slightly better than normal `0.003167`.
- [x] Installed ManiSkill3 into isolated venv `/root/autodl-tmp/envs/maniskill3-venv` without modifying `depthvla`.
- [x] ManiSkill3 `PushCube-v1` smoke passed for `state` and `pointcloud` observations. HDF5 outputs:
  - `experiments/logs/maniskill_pushcube_state_smoke.hdf5`
  - `experiments/logs/maniskill_pushcube_pointcloud_smoke.hdf5`
- [x] Replayed ManiSkill3 official `PushCube-v1` and `PickCube-v1` motion-planning demos into pointcloud observations with `pd_ee_delta_pos` actions.
- [x] Added `summarize_demo_hdf5.py`, generalized `validate_maniskill_hdf5.py`, and added `train_pointcloud_action_decoder.py`.
- [x] Ran the final point-cloud decoder pilot:
  - `PushCube-v1`: `20` demos, `1371` transitions, strict gate `3/3` seeds, mean paired normal-vs-cross L2 `0.002041`.
  - `PickCube-v1`: `20` demos, `1493` transitions, strict gate `2/3` seeds, mean paired normal-vs-cross L2 `0.022263`.
  - Interpretation: promising offline depth/action sensitivity, but not a final RGB-D success claim.
- [x] Added checkpoint saving and `eval_pointcloud_action_decoder.py` closed-loop smoke.
- [x] Ran PickCube closed-loop smoke for the tiny decoder:
  - pointcloud 1200-step normal/null/cross_demo: all `0/3`.
  - proprio 1200-step null: `0/3`.
  - pointcloud 5000-step normal/null/cross_demo: all `0/3`.
  - Interpretation: single-step PointNet BC has offline pointcloud-action coupling, but no reliable closed-loop success.
- [x] Added and ran a minimal 8-step PickCube action chunk decoder.
  - Offline gate passed with paired normal-vs-cross step L2 `0.027554`.
  - Closed-loop normal/null/cross_demo all `0/3`; normal and cross_demo rewards are tied.
  - Interpretation: shallow action chunk is also not enough; need stronger temporal/diffusion policy.
- [x] Added and ran `eval_pointcloud_geometry_controller.py` as a geometry feasibility probe.
  - PickCube normal pointcloud: `7/10`, mean reward `26.38`.
  - PickCube null pointcloud: `0/10`, mean reward `5.84`.
  - PickCube cross_demo pointcloud: `0/10`, mean reward `9.23`.
  - Interpretation: true pointcloud geometry is causally useful for closed-loop success; learned decoders have not captured the geometry-to-action policy.
- [x] Added and ran goal-conditioned and object-centric ManiSkill3 final attempts.
  - Goal-conditioned PointNet: normal RMSE `0.196182`, null `0.206774`, cross_sample `0.195966`; closed-loop normal/null/cross all `0/3`.
  - Object-centric feature MLP: offline gate passed with paired normal-vs-cross L2 `1.314387`, but closed-loop normal/null/cross all `0/10`.
  - Geometry controller with 150-step horizon and last-cube memory: normal `8/10`, null `0/10`, cross_demo `1/10`.
  - Interpretation: object geometry can be made causally visible to action prediction and explicit geometry control can solve the task, but shallow learned BC still lacks temporal control.
- [x] Added and ran geometry-teacher distillation.
  - No-phase teacher policy: normal `0/10` but mean reward `54.56`; debug showed grasp without stable move-goal.
  - Phase-conditioned teacher policy: normal `7/10`, null `0/10`, cross_demo `0/10` on 10 episodes.
  - 30-episode confirmation: normal `17/30`, null `0/30`, cross_demo `0/30`.
  - Phase/geometry disentanglement: normal geometry + normal phase `6/10`; null geometry + normal phase, cross geometry + normal phase, and normal geometry + null phase all `0/10`.
  - Interpretation: learned action decoder can use real pointcloud geometry if temporal phase is exposed; next step is learning phase/temporal state end-to-end.

## Blocked

- None currently. No training/eval process is running.

## Human Review Needed

- Default decision after method search and scaled ManiSkill3 runs: it is valid to claim ManiSkill3 PickCube pointcloud gain over matched no-depth baselines (`20/60` vs `1/60` / `3/60`) and eval-time corrupt controls (`1/60` / `1/60`). Do not claim OpenVLA-OFT end-to-end RGB-D > RGB-only.
- Human should choose whether the next step after submission is DP3-style diffusion action decoding, ACT-style temporal aggregation, recurrent phase inference, DAgger/distillation from the geometry controller, or an Act3D/PerAct-style 3D action-map classifier. `PickCube-v1` remains the best current first task. For OpenVLA/RLBench specifically, the next step should not be another shallow residual; it should add temporal/contact supervision or a stronger action decoder.
