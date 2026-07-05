#!/usr/bin/env python3
"""Collect final DepthVLA-OFT evidence tables from local JSON logs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any) -> str:
    if value is None:
        return "MISSING"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.3f}"
        if abs(value) >= 1:
            return f"{value:.6f}"
        if value == 0:
            return "0"
        return f"{value:.6g}"
    return str(value)


def add_row(rows: list[dict[str, str]], section: str, item: str, metric: str, value: Any, source: Path) -> None:
    rows.append(
        {
            "section": section,
            "item": item,
            "metric": metric,
            "value": fmt(value),
            "source": str(source.relative_to(REPO_ROOT)),
        }
    )


def overall_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """Handle both old diagnostic JSONs with an `overall` object and flat JSONs."""
    overall = data.get("overall")
    return overall if isinstance(overall, dict) else data


def collect_heatmap(rows: list[dict[str, str]]) -> None:
    specs = [
        ("open_drawer", REPO_ROOT / "experiments/logs/rlbench_projected_keypose_heatmap_probe_open_drawer.json"),
        ("stable6", REPO_ROOT / "experiments/logs/rlbench_projected_keypose_heatmap_probe_stable6.json"),
    ]
    for name, path in specs:
        data = load_json(path)
        result = (data or {}).get("result", {})
        for mode in ("normal", "cross_sample", "null"):
            add_row(
                rows,
                "offline_depth_signal",
                f"{name}:{mode}",
                "peak_error_px",
                result.get(mode, {}).get("peak_error_px"),
                path,
            )
        paired = result.get("paired_delta", {})
        add_row(
            rows,
            "offline_depth_signal",
            f"{name}:normal_vs_cross",
            "paired_peak_l2_px",
            paired.get("normal_vs_cross_peak_l2"),
            path,
        )


def collect_waypoint_diagnostic(rows: list[dict[str, str]]) -> None:
    path = REPO_ROOT / "experiments/logs/rlbench_open_drawer_waypoint_action/rlbench_policy_action_diag_rgbd_normal.json"
    data = load_json(path) or {}
    overall = overall_metrics(data)
    for metric in (
        "paired_pred_l1",
        "paired_pred_rmse",
        "paired_pred_xyz_l2",
        "xyz_rmse",
        "xyz_direction_cosine",
    ):
        add_row(rows, "waypoint_paired_diagnostic", "normal_vs_cross_sample", metric, overall.get(metric), path)


def collect_strict_diagnostics(rows: list[dict[str, str]]) -> None:
    specs = [
        ("normal", REPO_ROOT / "experiments/logs/rlbench_policy_action_diag_rgbd_normal.json"),
        ("null", REPO_ROOT / "experiments/logs/rlbench_policy_action_diag_rgbd_null.json"),
        ("cross_sample", REPO_ROOT / "experiments/logs/rlbench_policy_action_diag_rgbd_cross_sample.json"),
    ]
    for mode, path in specs:
        data = load_json(path) or {}
        overall = overall_metrics(data)
        for metric in ("rmse", "xyz_rmse", "xyz_direction_cosine", "gripper_abs_error"):
            add_row(rows, "waypoint_strict_diagnostic", mode, metric, overall.get(metric), path)


def collect_farthest_future_small(rows: list[dict[str, str]]) -> None:
    paired_path = REPO_ROOT / "experiments/logs/rlbench_farthest_future_small/rlbench_policy_action_diag_rgbd_normal.json"
    paired = overall_metrics(load_json(paired_path) or {})
    for metric in (
        "paired_pred_l1",
        "paired_pred_rmse",
        "paired_pred_xyz_l2",
        "xyz_rmse",
        "xyz_direction_cosine",
    ):
        add_row(
            rows,
            "farthest_future_500step_paired_diagnostic",
            "normal_vs_cross_sample",
            metric,
            paired.get(metric),
            paired_path,
        )

    strict_base = REPO_ROOT / "experiments/logs/rlbench_farthest_future_small_strict"
    for mode in ("normal", "null", "cross_sample"):
        path = strict_base / f"rlbench_policy_action_diag_rgbd_{mode}.json"
        overall = overall_metrics(load_json(path) or {})
        for metric in ("rmse", "xyz_rmse", "xyz_direction_cosine", "gripper_abs_error"):
            add_row(rows, "farthest_future_500step_strict_diagnostic", mode, metric, overall.get(metric), path)


def collect_farthest_future_5k(rows: list[dict[str, str]]) -> None:
    paired_path = REPO_ROOT / "experiments/logs/rlbench_farthest_future_5k/rlbench_policy_action_diag_rgbd_normal.json"
    paired = overall_metrics(load_json(paired_path) or {})
    for metric in (
        "paired_pred_l1",
        "paired_pred_rmse",
        "paired_pred_xyz_l2",
        "xyz_rmse",
        "xyz_direction_cosine",
    ):
        add_row(
            rows,
            "farthest_future_5k_paired_diagnostic",
            "normal_vs_cross_sample",
            metric,
            paired.get(metric),
            paired_path,
        )

    strict_base = REPO_ROOT / "experiments/logs/rlbench_farthest_future_5k_strict"
    for mode in ("normal", "null", "cross_sample"):
        path = strict_base / f"rlbench_policy_action_diag_rgbd_{mode}.json"
        overall = overall_metrics(load_json(path) or {})
        for metric in ("rmse", "xyz_rmse", "xyz_direction_cosine", "gripper_abs_error"):
            add_row(rows, "farthest_future_5k_strict_diagnostic", mode, metric, overall.get(metric), path)


def collect_visible_preclose_gate(rows: list[dict[str, str]]) -> None:
    feasibility_path = REPO_ROOT / "experiments/logs/rlbench_visible_contact_target_visible_preclose_probe.json"
    feasibility = load_json(feasibility_path) or {}
    result = feasibility.get("result", {})
    for mode in ("normal", "cross_sample", "ee_fallback"):
        mode_result = result.get(mode, {})
        item = f"open_drawer:visible_pre_first_close_point:{mode}"
        for metric in (
            "mean_m",
            "median_m",
            "p90_m",
            "within_0.020m",
            "within_0.050m",
        ):
            add_row(rows, "openvla_visible_preclose_feasibility", item, metric, mode_result.get(metric), feasibility_path)
    for metric in (
        "advantage_over_cross_median_m",
        "advantage_over_ee_median_m",
        "paired_normal_vs_cross_l1_m",
    ):
        add_row(
            rows,
            "openvla_visible_preclose_feasibility",
            "open_drawer:visible_pre_first_close_point",
            metric,
            result.get(metric),
            feasibility_path,
        )

    paired_path = REPO_ROOT / "experiments/logs/visible_preclose_500_paired/rlbench_policy_action_diag_rgbd_normal.json"
    paired = overall_metrics(load_json(paired_path) or {})
    for metric in (
        "depth_point_to_aux_label_l2",
        "compare_depth_point_to_aux_label_l2",
        "paired_depth_point_aux_label_l2_advantage",
        "paired_depth_point_xyz_l2",
        "paired_depth_waypoint_chunk_xyz_action_l2",
        "paired_pred_xyz_l2",
        "xyz_rmse",
        "xyz_direction_cosine",
    ):
        add_row(rows, "openvla_visible_preclose_paired_gate", "normal_vs_cross_sample", metric, paired.get(metric), paired_path)

    strict_base = REPO_ROOT / "experiments/logs/visible_preclose_500_strict"
    for mode in ("normal", "null", "cross_sample"):
        path = strict_base / f"rlbench_policy_action_diag_rgbd_{mode}.json"
        overall = overall_metrics(load_json(path) or {})
        for metric in (
            "xyz_rmse",
            "xyz_direction_cosine",
            "pred_xyz_norm",
            "depth_point_to_aux_label_l2",
            "depth_waypoint_chunk_xyz_action_norm",
        ):
            add_row(rows, "openvla_visible_preclose_strict_gate", mode, metric, overall.get(metric), path)


def collect_rollout(rows: list[dict[str, str]]) -> None:
    base = REPO_ROOT / "experiments/logs/rlbench_open_drawer_waypoint_action/eval_h200"
    for mode in ("normal", "null", "cross_sample"):
        path = base / f"rgbd_{mode}.json"
        data = load_json(path) or {}
        task = data.get("task_results", {}).get("open_drawer", {})
        add_row(rows, "waypoint_rollout", mode, "success_rate", data.get("success_rate"), path)
        add_row(rows, "waypoint_rollout", mode, "length", task.get("length"), path)
        add_row(rows, "waypoint_rollout", mode, "errors", task.get("errors", {}), path)
        add_row(rows, "waypoint_rollout", mode, "delta_xyz_norm_mean", task.get("delta_xyz_norm_mean"), path)


def collect_action_map_feasibility(rows: list[dict[str, str]]) -> None:
    paths = sorted((REPO_ROOT / "experiments/logs").glob("rlbench_3d_action_map_feasibility*.json"))
    for path in paths:
        data = load_json(path) or {}
        config = data.get("config", {})
        item = f"{path.stem.replace('rlbench_3d_action_map_feasibility_', '')}:{config.get('target', 'unknown')}"
        result = data.get("result", {})
        for mode in ("normal", "cross_sample", "ee_fallback"):
            row = result.get(mode, {})
            add_row(rows, "action_map_candidate_feasibility", f"{item}:{mode}", "median_m", row.get("median_m"), path)
            add_row(rows, "action_map_candidate_feasibility", f"{item}:{mode}", "p90_m", row.get("p90_m"), path)
        add_row(
            rows,
            "action_map_candidate_feasibility",
            item,
            "advantage_over_cross_median_m",
            result.get("advantage_over_cross_median_m"),
            path,
        )
        add_row(
            rows,
            "action_map_candidate_feasibility",
            item,
            "advantage_over_ee_median_m",
            result.get("advantage_over_ee_median_m"),
            path,
        )


def seed_from_path(path: Path) -> str:
    stem = path.stem
    if "seed" not in stem:
        return "7"
    return stem.rsplit("seed", 1)[-1]


def collect_maniskill_pilot(rows: list[dict[str, str]]) -> None:
    specs = [
        (
            "pushcube",
            "pointcloud_train",
            "maniskill_pushcube_pointcloud_decoder_gate_20demo_strictcross*.json",
        ),
        (
            "pushcube",
            "proprio_null_train",
            "maniskill_pushcube_proprio_decoder_gate_20demo_seed*.json",
        ),
        (
            "pickcube",
            "pointcloud_train",
            "maniskill_pickcube_pointcloud_decoder_gate_20demo_strictcross_seed*.json",
        ),
        (
            "pickcube",
            "proprio_null_train",
            "maniskill_pickcube_proprio_decoder_gate_20demo_seed*.json",
        ),
    ]
    for task, family, pattern in specs:
        paths = [
            path
            for path in sorted((REPO_ROOT / "experiments/logs").glob(pattern))
            if "_ckpt" not in path.stem and "_5k" not in path.stem
        ]
        values: dict[str, list[float]] = {
            "normal_raw_rmse": [],
            "null_raw_rmse": [],
            "cross_sample_raw_rmse": [],
            "paired_normal_vs_cross_l2": [],
        }
        pass_count = 0
        for path in paths:
            data = load_json(path) or {}
            metrics = data.get("metrics", {})
            deltas = data.get("paired_deltas", {})
            gate = data.get("gate", {})
            seed = seed_from_path(path)
            item = f"{task}:{family}:seed{seed}"
            normal_rmse = metrics.get("normal", {}).get("raw_rmse")
            null_rmse = metrics.get("null", {}).get("raw_rmse")
            cross_rmse = metrics.get("cross_sample", {}).get("raw_rmse")
            delta_l2 = deltas.get("paired_normal_vs_cross_sample_l2")
            passed = bool(gate.get("passed", False))

            add_row(rows, "maniskill_pointcloud_decoder_pilot", item, "normal_raw_rmse", normal_rmse, path)
            add_row(rows, "maniskill_pointcloud_decoder_pilot", item, "null_raw_rmse", null_rmse, path)
            add_row(rows, "maniskill_pointcloud_decoder_pilot", item, "cross_sample_raw_rmse", cross_rmse, path)
            add_row(
                rows,
                "maniskill_pointcloud_decoder_pilot",
                item,
                "paired_normal_vs_cross_l2",
                delta_l2,
                path,
            )
            add_row(rows, "maniskill_pointcloud_decoder_pilot", item, "gate_passed", passed, path)

            if isinstance(normal_rmse, (float, int)):
                values["normal_raw_rmse"].append(float(normal_rmse))
            if isinstance(null_rmse, (float, int)):
                values["null_raw_rmse"].append(float(null_rmse))
            if isinstance(cross_rmse, (float, int)):
                values["cross_sample_raw_rmse"].append(float(cross_rmse))
            if isinstance(delta_l2, (float, int)):
                values["paired_normal_vs_cross_l2"].append(float(delta_l2))
            pass_count += int(passed)

        if paths:
            summary_item = f"{task}:{family}:summary"
            for metric, metric_values in values.items():
                add_row(
                    rows,
                    "maniskill_pointcloud_decoder_pilot",
                    summary_item,
                    f"mean_{metric}",
                    statistics.mean(metric_values) if metric_values else None,
                    paths[0],
                )
            add_row(
                rows,
                "maniskill_pointcloud_decoder_pilot",
                summary_item,
                "pass_count",
                f"{pass_count}/{len(paths)}",
                paths[0],
            )


def collect_maniskill_rollout_smoke(rows: list[dict[str, str]]) -> None:
    paths = sorted((REPO_ROOT / "experiments/logs").glob("maniskill_pickcube_rollout_*_seed7*.json"))
    for path in paths:
        if "learned_cube_controller" in path.stem:
            continue
        data = load_json(path) or {}
        item = path.stem.replace("maniskill_pickcube_rollout_", "")
        for metric in ("success_rate", "mean_length", "mean_reward_sum"):
            add_row(rows, "maniskill_closed_loop_smoke", item, metric, data.get(metric), path)


def collect_maniskill_chunk_gate(rows: list[dict[str, str]]) -> None:
    path = REPO_ROOT / "experiments/logs/maniskill_pickcube_pointcloud_chunk_gate_h8_seed7_5k.json"
    data = load_json(path)
    if data is None:
        return
    item = "pickcube:pointcloud_chunk_h8_seed7_5k"
    metrics = data.get("metrics", {})
    deltas = data.get("paired_deltas", {})
    gate = data.get("gate", {})
    add_row(rows, "maniskill_chunk_decoder_pilot", item, "normal_raw_rmse", metrics.get("normal", {}).get("raw_rmse"), path)
    add_row(rows, "maniskill_chunk_decoder_pilot", item, "null_raw_rmse", metrics.get("null", {}).get("raw_rmse"), path)
    add_row(
        rows,
        "maniskill_chunk_decoder_pilot",
        item,
        "cross_sample_raw_rmse",
        metrics.get("cross_sample", {}).get("raw_rmse"),
        path,
    )
    add_row(
        rows,
        "maniskill_chunk_decoder_pilot",
        item,
        "paired_normal_vs_cross_step_l2",
        deltas.get("paired_normal_vs_cross_sample_step_l2"),
        path,
    )
    add_row(rows, "maniskill_chunk_decoder_pilot", item, "gate_passed", gate.get("passed"), path)


def collect_maniskill_extra_gates(rows: list[dict[str, str]]) -> None:
    specs = [
        (
            "maniskill_goal_conditioned_decoder",
            "pickcube:goal_pointcloud_seed7_5k",
            REPO_ROOT / "experiments/logs/maniskill_pickcube_goal_pointcloud_gate_seed7_5k.json",
            "paired_normal_vs_cross_sample_l2",
        ),
        (
            "maniskill_object_feature_decoder",
            "pickcube:object_feature_seed7_5k",
            REPO_ROOT / "experiments/logs/maniskill_pickcube_object_feature_gate_seed7_5k.json",
            "paired_normal_vs_cross_sample_l2",
        ),
    ]
    for section, item, path, delta_key in specs:
        data = load_json(path)
        if data is None:
            continue
        metrics = data.get("metrics", {})
        deltas = data.get("paired_deltas", {})
        gate = data.get("gate", {})
        add_row(rows, section, item, "normal_raw_rmse", metrics.get("normal", {}).get("raw_rmse"), path)
        add_row(rows, section, item, "null_raw_rmse", metrics.get("null", {}).get("raw_rmse"), path)
        add_row(rows, section, item, "cross_sample_raw_rmse", metrics.get("cross_sample", {}).get("raw_rmse"), path)
        add_row(rows, section, item, "paired_normal_vs_cross_l2", deltas.get(delta_key), path)
        add_row(rows, section, item, "gate_passed", gate.get("passed"), path)


def collect_maniskill_teacher_distillation(rows: list[dict[str, str]]) -> None:
    specs = [
        (
            "pickcube:teacher_object_feature_success30_seed7_5k",
            REPO_ROOT / "experiments/logs/maniskill_pickcube_teacher_object_feature_gate_success30_seed7_5k.json",
        ),
        (
            "pickcube:teacher_object_feature_phase_success30_seed7_5k",
            REPO_ROOT / "experiments/logs/maniskill_pickcube_teacher_object_feature_phase_gate_success30_seed7_5k.json",
        ),
    ]
    for item, path in specs:
        data = load_json(path)
        if data is None:
            continue
        for split in ("train", "val"):
            metrics = data.get("metrics", {}).get(split, {})
            add_row(rows, "maniskill_teacher_distillation", f"{item}:{split}", "raw_rmse", metrics.get("raw_rmse"), path)
            add_row(rows, "maniskill_teacher_distillation", f"{item}:{split}", "raw_l1", metrics.get("raw_l1"), path)
        add_row(rows, "maniskill_teacher_distillation", item, "num_samples", data.get("num_samples"), path)
        add_row(rows, "maniskill_teacher_distillation", item, "num_episodes", data.get("num_episodes"), path)


def collect_maniskill_learned_phase(rows: list[dict[str, str]]) -> None:
    classifier_path = REPO_ROOT / "experiments/logs/maniskill_pickcube_teacher_phase_classifier_success30_seed7_5k.json"
    classifier = load_json(classifier_path)
    if classifier is not None:
        for split in ("train", "val"):
            metrics = classifier.get("metrics", {}).get(split, {})
            item = f"pickcube:phase_classifier:{split}"
            for metric in ("accuracy", "macro_accuracy", "cross_entropy", "mean_confidence"):
                add_row(rows, "maniskill_learned_phase", item, metric, metrics.get(metric), classifier_path)

    rollout_specs = [
        (
            "pickcube:learned_phase_normal:30eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_object_feature_learnedphase_normal_success30_seed7_5k_150steps_30eps.json",
        ),
        (
            "pickcube:learned_phase_null:30eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_object_feature_learnedphase_null_success30_seed7_5k_150steps_30eps.json",
        ),
        (
            "pickcube:learned_phase_cross_demo:30eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_object_feature_learnedphase_cross_demo_success30_seed7_5k_150steps_30eps.json",
        ),
        (
            "pickcube:learned_phase_nullgeom_normalphase:10eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_object_feature_learnedphase_nullgeom_normalphase_seed7_150steps_10eps.json",
        ),
        (
            "pickcube:learned_phase_crossgeom_normalphase:10eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_object_feature_learnedphase_crossgeom_normalphase_seed7_150steps_10eps.json",
        ),
        (
            "pickcube:learned_phase_normalgeom_nullphase:10eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_object_feature_learnedphase_normalgeom_nullphase_seed7_150steps_10eps.json",
        ),
    ]
    for item, path in rollout_specs:
        data = load_json(path)
        if data is None:
            continue
        episodes = data.get("episodes", [])
        success_count = sum(1 for episode in episodes if episode.get("success"))
        total = len(episodes)
        add_row(rows, "maniskill_learned_phase", item, "success_count", f"{success_count}/{total}", path)
        for metric in ("success_rate", "mean_length", "mean_reward_sum"):
            add_row(rows, "maniskill_learned_phase", item, metric, data.get(metric), path)


def collect_maniskill_raw_pointcloud_teacher(rows: list[dict[str, str]]) -> None:
    gate_path = REPO_ROOT / "experiments/logs/maniskill_pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success30_seed7_5k.json"
    gate = load_json(gate_path)
    if gate is not None:
        for mode in ("normal", "null", "cross_sample"):
            metrics = gate.get("metrics", {}).get(mode, {})
            item = f"pickcube:raw_pointcloud_cropz002:{mode}"
            for metric in ("raw_rmse", "raw_l1", "phase_accuracy", "cube_xyz_rmse"):
                add_row(rows, "maniskill_raw_pointcloud_teacher", item, metric, metrics.get(metric), gate_path)
        deltas = gate.get("paired_deltas", {})
        for metric in (
            "paired_normal_vs_cross_sample_l2",
            "paired_normal_vs_null_l2",
            "paired_normal_vs_cross_sample_phase_change_rate",
        ):
            add_row(
                rows,
                "maniskill_raw_pointcloud_teacher",
                "pickcube:raw_pointcloud_cropz002:paired",
                metric,
                deltas.get(metric),
                gate_path,
            )
        add_row(
            rows,
            "maniskill_raw_pointcloud_teacher",
            "pickcube:raw_pointcloud_cropz002:gate",
            "passed",
            gate.get("gate", {}).get("passed"),
            gate_path,
        )

    rollout_specs = [
        (
            "pickcube:raw_pointcloud_cropz002_normal:30eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_normal_success30_seed7_5k_150steps_30eps.json",
        ),
        (
            "pickcube:raw_pointcloud_cropz002_null:30eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_null_success30_seed7_5k_150steps_30eps.json",
        ),
        (
            "pickcube:raw_pointcloud_cropz002_cross_demo:30eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_cross_demo_success30_seed7_5k_150steps_30eps.json",
        ),
    ]
    for item, path in rollout_specs:
        data = load_json(path)
        if data is None:
            continue
        episodes = data.get("episodes", [])
        success_count = sum(1 for episode in episodes if episode.get("success"))
        add_row(rows, "maniskill_raw_pointcloud_teacher", item, "success_count", f"{success_count}/{len(episodes)}", path)
        for metric in ("success_rate", "mean_length", "mean_reward_sum"):
            add_row(rows, "maniskill_raw_pointcloud_teacher", item, metric, data.get(metric), path)

    scaled_gate_path = (
        REPO_ROOT
        / "experiments/logs/maniskill_pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_seed7_10k_h256.json"
    )
    scaled_gate = load_json(scaled_gate_path)
    if scaled_gate is not None:
        for mode in ("normal", "null", "cross_sample"):
            metrics = scaled_gate.get("metrics", {}).get(mode, {})
            item = f"pickcube:raw_pointcloud_cropz002_success100_h256_10k:{mode}"
            for metric in ("raw_rmse", "raw_l1", "phase_accuracy", "cube_xyz_rmse"):
                add_row(rows, "maniskill_raw_pointcloud_teacher_scaled", item, metric, metrics.get(metric), scaled_gate_path)
        deltas = scaled_gate.get("paired_deltas", {})
        for metric in (
            "paired_normal_vs_cross_sample_l2",
            "paired_normal_vs_null_l2",
            "paired_normal_vs_cross_sample_phase_change_rate",
        ):
            add_row(
                rows,
                "maniskill_raw_pointcloud_teacher_scaled",
                "pickcube:raw_pointcloud_cropz002_success100_h256_10k:paired",
                metric,
                deltas.get(metric),
                scaled_gate_path,
            )
        add_row(
            rows,
            "maniskill_raw_pointcloud_teacher_scaled",
            "pickcube:raw_pointcloud_cropz002_success100_h256_10k:gate",
            "passed",
            scaled_gate.get("gate", {}).get("passed"),
            scaled_gate_path,
        )

    scaled_rollout_specs = [
        (
            "pickcube:raw_pointcloud_success100_normal:seed4100_30eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_normal_seed7_150steps_30eps.json",
        ),
        (
            "pickcube:raw_pointcloud_success100_null:seed4100_30eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_null_seed7_150steps_30eps.json",
        ),
        (
            "pickcube:raw_pointcloud_success100_cross_demo:seed4100_30eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_cross_demo_seed7_150steps_30eps.json",
        ),
        (
            "pickcube:raw_pointcloud_success100_normal:seed4500_30eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_normal_seed7_150steps_seed4500_30eps.json",
        ),
        (
            "pickcube:raw_pointcloud_success100_null:seed4500_30eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_null_seed7_150steps_seed4500_30eps.json",
        ),
        (
            "pickcube:raw_pointcloud_success100_cross_demo:seed4500_30eps",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_cross_demo_seed7_150steps_seed4500_30eps.json",
        ),
    ]
    aggregate: dict[str, dict[str, Any]] = {
        "normal": {"success": 0, "total": 0, "reward": []},
        "null": {"success": 0, "total": 0, "reward": []},
        "cross_demo": {"success": 0, "total": 0, "reward": []},
    }
    for item, path in scaled_rollout_specs:
        data = load_json(path)
        if data is None:
            continue
        episodes = data.get("episodes", [])
        success_count = sum(1 for episode in episodes if episode.get("success"))
        add_row(
            rows,
            "maniskill_raw_pointcloud_teacher_scaled",
            item,
            "success_count",
            f"{success_count}/{len(episodes)}",
            path,
        )
        for metric in ("success_rate", "mean_length", "mean_reward_sum"):
            add_row(rows, "maniskill_raw_pointcloud_teacher_scaled", item, metric, data.get(metric), path)
        if "_normal_" in path.stem:
            mode = "normal"
        elif "_null_" in path.stem:
            mode = "null"
        else:
            mode = "cross_demo"
        aggregate[mode]["success"] += success_count
        aggregate[mode]["total"] += len(episodes)
        if isinstance(data.get("mean_reward_sum"), (float, int)):
            aggregate[mode]["reward"].append(float(data["mean_reward_sum"]))

    for mode, values in aggregate.items():
        if not values["total"]:
            continue
        item = f"pickcube:raw_pointcloud_success100_{mode}:aggregate_60eps"
        add_row(
            rows,
            "maniskill_raw_pointcloud_teacher_scaled",
            item,
            "success_count",
            f"{values['success']}/{values['total']}",
            scaled_rollout_specs[0][1],
        )
        add_row(
            rows,
            "maniskill_raw_pointcloud_teacher_scaled",
            item,
            "success_rate",
            values["success"] / values["total"],
            scaled_rollout_specs[0][1],
        )
        add_row(
            rows,
            "maniskill_raw_pointcloud_teacher_scaled",
            item,
            "mean_reward_sum_across_eval_seeds",
            statistics.mean(values["reward"]) if values["reward"] else None,
            scaled_rollout_specs[0][1],
        )

    scaled_chunk_path = (
        REPO_ROOT
        / "experiments/logs/maniskill_pickcube_teacher_pointcloud_cropz002_action_chunk_h8_cubeaux_success100_seed7_10k_h256.json"
    )
    scaled_chunk = load_json(scaled_chunk_path)
    if scaled_chunk is not None:
        metrics = scaled_chunk.get("metrics", {})
        deltas = scaled_chunk.get("paired_deltas", {})
        item = "pickcube:raw_pointcloud_chunk_h8_success100_h256_10k"
        for mode in ("normal", "null", "cross_sample"):
            add_row(
                rows,
                "maniskill_raw_pointcloud_chunk_scaled",
                f"{item}:{mode}",
                "raw_rmse",
                metrics.get(mode, {}).get("raw_rmse"),
                scaled_chunk_path,
            )
        add_row(
            rows,
            "maniskill_raw_pointcloud_chunk_scaled",
            item,
            "paired_normal_vs_cross_step_l2",
            deltas.get("paired_normal_vs_cross_sample_step_l2"),
            scaled_chunk_path,
        )
        add_row(
            rows,
            "maniskill_raw_pointcloud_chunk_scaled",
            item,
            "gate_passed",
            scaled_chunk.get("gate", {}).get("passed"),
            scaled_chunk_path,
        )

    baseline_specs = [
        (
            "rgb_only_train",
            "rgb_only",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_rgbonly_seed7_10k_h256.json",
            [
                REPO_ROOT
                / "experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_rgbonly_seed7_150steps_30eps.json",
                REPO_ROOT
                / "experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_rgbonly_seed7_150steps_seed4500_30eps.json",
            ],
        ),
        (
            "null_train",
            "null",
            REPO_ROOT
            / "experiments/logs/maniskill_pickcube_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_nulltrain_seed7_10k_h256.json",
            [
                REPO_ROOT
                / "experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_nulltrain_seed7_150steps_30eps.json",
                REPO_ROOT
                / "experiments/logs/maniskill_pickcube_rollout_teacher_pointcloud_cropz002_phase_action_cubeaux_success100_nulltrain_seed7_150steps_seed4500_30eps.json",
            ],
        ),
    ]
    for baseline_name, eval_mode, gate_path, rollout_paths in baseline_specs:
        gate = load_json(gate_path)
        if gate is not None:
            metrics = gate.get("metrics", {}).get(eval_mode, {})
            item = f"pickcube:{baseline_name}:offline"
            for metric in ("raw_rmse", "raw_l1", "phase_accuracy", "cube_xyz_rmse"):
                add_row(rows, "maniskill_matched_no_depth_baselines", item, metric, metrics.get(metric), gate_path)
            add_row(
                rows,
                "maniskill_matched_no_depth_baselines",
                item,
                "train_point_mode",
                gate.get("train_point_mode"),
                gate_path,
            )
        total_success = 0
        total_episodes = 0
        rewards: list[float] = []
        for path in rollout_paths:
            data = load_json(path)
            if data is None:
                continue
            episodes = data.get("episodes", [])
            success_count = sum(1 for episode in episodes if episode.get("success"))
            total_success += success_count
            total_episodes += len(episodes)
            if isinstance(data.get("mean_reward_sum"), (float, int)):
                rewards.append(float(data["mean_reward_sum"]))
            eval_seed = "seed4500" if "seed4500" in path.stem else "seed4100"
            item = f"pickcube:{baseline_name}:{eval_seed}_30eps"
            add_row(
                rows,
                "maniskill_matched_no_depth_baselines",
                item,
                "success_count",
                f"{success_count}/{len(episodes)}",
                path,
            )
            for metric in ("success_rate", "mean_length", "mean_reward_sum"):
                add_row(rows, "maniskill_matched_no_depth_baselines", item, metric, data.get(metric), path)
        if total_episodes:
            item = f"pickcube:{baseline_name}:aggregate_60eps"
            add_row(
                rows,
                "maniskill_matched_no_depth_baselines",
                item,
                "success_count",
                f"{total_success}/{total_episodes}",
                rollout_paths[0],
            )
            add_row(
                rows,
                "maniskill_matched_no_depth_baselines",
                item,
                "success_rate",
                total_success / total_episodes,
                rollout_paths[0],
            )
            add_row(
                rows,
                "maniskill_matched_no_depth_baselines",
                item,
                "mean_reward_sum_across_eval_seeds",
                statistics.mean(rewards) if rewards else None,
                rollout_paths[0],
            )

    controller_specs = [
        (
            "pickcube:learned_cube_controller_normal:30eps",
            REPO_ROOT / "experiments/logs/maniskill_pickcube_rollout_learned_cube_controller_cropz002_normal_seed7_150steps_30eps.json",
        ),
        (
            "pickcube:learned_cube_controller_null:30eps",
            REPO_ROOT / "experiments/logs/maniskill_pickcube_rollout_learned_cube_controller_cropz002_null_seed7_150steps_30eps.json",
        ),
        (
            "pickcube:learned_cube_controller_cross_demo:30eps",
            REPO_ROOT / "experiments/logs/maniskill_pickcube_rollout_learned_cube_controller_cropz002_cross_demo_seed7_150steps_30eps.json",
        ),
    ]
    for item, path in controller_specs:
        data = load_json(path)
        if data is None:
            continue
        episodes = data.get("episodes", [])
        success_count = sum(1 for episode in episodes if episode.get("success"))
        add_row(rows, "maniskill_learned_cube_controller", item, "success_count", f"{success_count}/{len(episodes)}", path)
        for metric in ("success_rate", "mean_length", "mean_reward_sum"):
            add_row(rows, "maniskill_learned_cube_controller", item, metric, data.get(metric), path)


def collect_maniskill_geometry_controller(rows: list[dict[str, str]]) -> None:
    specs = [
        ("normal", REPO_ROOT / "experiments/logs/maniskill_pickcube_geometry_normal_seed4100_goalz_10eps.json"),
        ("null", REPO_ROOT / "experiments/logs/maniskill_pickcube_geometry_null_seed4100_goalz_10eps.json"),
        ("cross_demo", REPO_ROOT / "experiments/logs/maniskill_pickcube_geometry_cross_demo_seed4100_goalz_10eps.json"),
        (
            "normal_memory_150steps",
            REPO_ROOT / "experiments/logs/maniskill_pickcube_geometry_normal_seed4100_memory_150steps_10eps.json",
        ),
        (
            "null_memory_150steps",
            REPO_ROOT / "experiments/logs/maniskill_pickcube_geometry_null_seed4100_memory_150steps_10eps.json",
        ),
        (
            "cross_demo_memory_150steps",
            REPO_ROOT / "experiments/logs/maniskill_pickcube_geometry_cross_demo_seed4100_memory_150steps_10eps.json",
        ),
    ]
    for mode, path in specs:
        data = load_json(path)
        if data is None:
            continue
        item = f"pickcube_geometry:{mode}:10eps"
        for metric in ("success_rate", "mean_length", "mean_reward_sum"):
            add_row(rows, "maniskill_geometry_controller", item, metric, data.get(metric), path)


def write_csv(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "item", "metric", "value", "source"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], output: Path, csv_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DepthVLA-OFT Final Results Table",
        "",
        "Generated from local JSON logs by `scripts/collect_depthvla_final_results.py`.",
        "",
        f"CSV: `{csv_path.relative_to(REPO_ROOT)}`",
        "",
        "| section | item | metric | value | source |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['section']} | {row['item']} | `{row['metric']}` | `{row['value']}` | `{row['source']}` |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Offline heatmap probes show real depth signal: normal depth has much lower peak error than null/cross-sample depth.",
            "- The final waypoint-action policy has only a tiny paired normal-vs-cross action delta.",
            "- The 500-step `farthest_future_pose_xyz` waypoint smoke also fails the causal gate: paired action delta is smaller than the previous waypoint run and normal/cross-sample strict diagnostics are effectively tied.",
            "- The 5000-step `farthest_future_pose_xyz` run still fails the causal gate: paired action delta shrinks further, and cross-sample slightly beats normal in strict `xyz_rmse`.",
            "- The latest OpenVLA/RLBench `visible_pre_first_close_point_xyz` gate gives the cleanest OpenVLA geometry evidence: normal selected-point-to-aux-label L2 is `0.099m`, better than null `0.699m` and cross-sample `0.194m`; the selected point also enters the 8-step waypoint/action chunk. It is still not rollout success, because strict action imitation does not beat cross-sample.",
            "- Older OpenVLA waypoint/farthest-future strict diagnostics are nearly tied and rollout fails for normal/null/cross-sample, so those recipes remain no-go for RGB-D task improvement.",
            "- 3D action-map candidate coverage shows the target-design issue directly: short-horizon keypose/next-pose labels have a strong EE-position shortcut, while future/final/farthest-future labels make normal point candidates beat both cross-sample candidates and EE fallback.",
            "- The early ManiSkill3 pilot was only an offline signal, but the later scaled PickCube run is a closed-loop learned-policy result: raw cropped pointcloud normal reaches `20/60`, while eval-time null/cross-demo reach only `1/60` each.",
            "- A follow-up ManiSkill3 closed-loop smoke still fails: the tiny single-step PointNet BC decoder gets `0/3` success on `PickCube-v1` even after 5000 steps, so the next implementation needs chunked/diffusion action decoding or stronger closed-loop training.",
            "- A minimal 8-step action chunk decoder also fails closed-loop (`0/3` normal), and normal does not beat cross-demo reward; a stronger temporal policy is still required.",
            "- Goal-conditioned PointNet still fails the strict offline gate and closed-loop smoke: adding `goal_pos` alone is not enough.",
            "- An object-centric feature MLP passes the offline causal gate strongly, but closed-loop remains `0/10`; the null condition is OOD for its `cube_valid` feature, so the most meaningful comparison is normal vs cross-sample. The failure has moved from perception to temporal control.",
            "- A pointcloud-geometry controller is a positive diagnostic on `PickCube-v1`: normal pointcloud reaches `7/10` at 100 steps and `8/10` at 150 steps, while null/cross controls stay far behind. This shows the task is RGB-D/pointcloud solvable, but the learned decoder has not captured the geometry-to-action algorithm yet.",
            "- The phase-conditioned teacher-distilled learned policy is a positive diagnostic: normal pointcloud reaches `17/30`, while null and cross-demo are `0/30`. This intermediate result still uses a hand-written phase state machine, but it shows learned actions can exploit true geometry when temporal state is exposed.",
            "- A phase/geometry disentanglement control confirms this is not phase alone: with the same seed, normal geometry + normal phase is `6/10`, while null geometry + normal phase, cross geometry + normal phase, and normal geometry + null phase are all `0/10`.",
            "- The learned phase follow-up removes the hand-written phase signal at rollout time: a single-frame phase classifier reaches `96.1%` validation accuracy and the learned-phase policy reaches normal `19/30`, null `0/30`, and cross-demo `0/30`. Learned-phase disentanglement is also strict: null geometry + learned normal phase, cross geometry + learned normal phase, and normal geometry + learned null phase are all `0/10`.",
            "- A raw cropped-pointcloud follow-up moves one step closer to RGB-D input: with `z>0.02` point sampling and a cube-center auxiliary head, the offline gate passes strongly (normal cube RMSE `0.009m`, cross/null about `0.075m`; paired normal-vs-cross action L2 `0.215`). Closed-loop remains weak but positive: normal `2/30`, null `0/30`, cross-demo `0/30`.",
            "- Scaling the raw cropped-pointcloud teacher data from 30 to 100 successful episodes and training a larger h256/10k single-step decoder turns the weak signal into a clear learned-policy result: across two 30-episode eval seeds, normal reaches `20/60`, while null and cross-demo are only `1/60` each. This is the strongest raw-pointcloud learned action result and the key positive evidence that depth/pointcloud geometry can become closed-loop benefit on a 3D-sensitive benchmark.",
            "- Matched no-depth training baselines strengthen that claim: with the same 100 teacher episodes, model capacity, and eval seeds, an `rgb_only` sampled-color baseline reaches only `1/60`, and a pure `null` train baseline reaches `3/60`. Normal pointcloud is therefore far ahead of both train-time no-depth baselines.",
            "- The scaled raw-pointcloud action-chunk variant passes the offline gate too, but does not improve success over the single-step decoder; this points back to temporal/action decoding quality rather than raw perception.",
            "- Using the same learned raw-pointcloud cube predictor inside the fixed geometry controller gives a much stronger diagnostic: normal `22/30`, null `1/30`, cross-demo `0/30`. This proves raw depth/pointcloud perception is strong enough for control; the remaining bottleneck is the learned action/temporal decoder.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", type=Path, default=REPO_ROOT / "FINAL_RESULTS_TABLE.md")
    parser.add_argument("--csv", type=Path, default=REPO_ROOT / "experiments/logs/final_results_table.csv")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    collect_heatmap(rows)
    collect_waypoint_diagnostic(rows)
    collect_strict_diagnostics(rows)
    collect_farthest_future_small(rows)
    collect_farthest_future_5k(rows)
    collect_visible_preclose_gate(rows)
    collect_rollout(rows)
    collect_action_map_feasibility(rows)
    collect_maniskill_pilot(rows)
    collect_maniskill_rollout_smoke(rows)
    collect_maniskill_chunk_gate(rows)
    collect_maniskill_extra_gates(rows)
    collect_maniskill_teacher_distillation(rows)
    collect_maniskill_learned_phase(rows)
    collect_maniskill_raw_pointcloud_teacher(rows)
    collect_maniskill_geometry_controller(rows)

    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown, args.csv)
    print(f"wrote {args.markdown}")
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
