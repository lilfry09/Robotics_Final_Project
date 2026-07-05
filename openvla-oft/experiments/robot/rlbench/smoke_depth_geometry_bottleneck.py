"""Smoke test for OpenVLA DepthVLA geometry-bottleneck action path.

This avoids the full training stack and checks the local modules that matter for
the next OpenVLA end-to-end attempt:

- dense point tokens carry aligned 3D point features
- ``visible_object_point_xyz`` and visible contact targets use point selection
  instead of pooled features
- waypoint action override routes the selected EE-relative geometry into action
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8


def load_action_head_class():
    constants = types.ModuleType("prismatic.vla.constants")
    constants.ACTION_DIM = ACTION_DIM
    constants.ACTION_TOKEN_BEGIN_IDX = 32000
    constants.IGNORE_INDEX = -100
    constants.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK
    constants.PROPRIO_DIM = 8
    constants.STOP_INDEX = 2
    sys.modules.setdefault("prismatic", types.ModuleType("prismatic"))
    sys.modules.setdefault("prismatic.vla", types.ModuleType("prismatic.vla"))
    sys.modules["prismatic.vla.constants"] = constants

    diffusers = types.ModuleType("diffusers")
    schedulers = types.ModuleType("diffusers.schedulers")
    scheduling_ddim = types.ModuleType("diffusers.schedulers.scheduling_ddim")

    class _DDIMSchedulerStub:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("DDIMScheduler is not needed by this L1 action-head smoke")

    scheduling_ddim.DDIMScheduler = _DDIMSchedulerStub
    sys.modules.setdefault("diffusers", diffusers)
    sys.modules.setdefault("diffusers.schedulers", schedulers)
    sys.modules["diffusers.schedulers.scheduling_ddim"] = scheduling_ddim

    spec = importlib.util.spec_from_file_location(
        "depthvla_action_heads_smoke",
        PROJECT_ROOT / "prismatic/models/action_heads.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.L1RegressionActionHead


def main() -> None:
    torch.manual_seed(0)
    batch_size = 2
    num_points = 32
    hidden_dim = 32

    L1RegressionActionHead = load_action_head_class()
    action_head = L1RegressionActionHead(
        input_dim=hidden_dim,
        hidden_dim=64,
        action_dim=ACTION_DIM,
        use_depth_conditioning=True,
        depth_fusion_type="object_query",
        depth_fusion_gate_init=1.0,
        depth_adapter_hidden_dim=64,
        spatial_aux_output_dim=3,
        depth_waypoint_action_weight=1.0,
        depth_waypoint_action_clip=0.05,
    )
    actions_hidden = torch.randn(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim)
    depth_context = torch.randn(batch_size, num_points, hidden_dim)
    point_features = torch.zeros(batch_size, num_points, 12)
    point_features[..., :3] = torch.randn(batch_size, num_points, 3) * 0.2 + torch.tensor([0.3, 0.0, 0.9])
    point_features[..., 3:6] = torch.randn(batch_size, num_points, 3) * 0.1
    point_features[..., 8] = 1.0

    selected_xyz = action_head.predict_spatial_delta(
        depth_context,
        actions_hidden_states=actions_hidden,
        aux_target="visible_object_point_xyz",
        depth_point_features=point_features,
    )
    assert selected_xyz.shape == (batch_size, 3), selected_xyz.shape
    visible_preclose_xyz = action_head.predict_spatial_delta(
        depth_context,
        actions_hidden_states=actions_hidden,
        aux_target="visible_pre_first_close_point_xyz",
        depth_point_features=point_features,
    )
    assert visible_preclose_xyz.shape == (batch_size, 3), visible_preclose_xyz.shape

    visible_rel = action_head.predict_spatial_delta(
        depth_context,
        actions_hidden_states=actions_hidden,
        aux_target="visible_object_rel_xyz",
        depth_point_features=point_features,
    )
    assert visible_rel.shape == (batch_size, 3), visible_rel.shape

    action = action_head.predict_action(
        actions_hidden,
        depth_context=depth_context,
        depth_point_features=point_features,
    )
    assert action.shape == (batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM), action.shape
    waypoint = action_head.last_depth_waypoint_xyz_action
    assert waypoint is not None
    assert waypoint.shape == (batch_size, 3), waypoint.shape
    assert torch.allclose(action[:, 0, :3].float(), waypoint.float(), atol=1e-5)
    waypoint_abs_max = float(action[:, 0, :3].detach().abs().max())
    assert waypoint_abs_max <= 0.0501

    print("Depth geometry bottleneck smoke passed")
    print("  selected_xyz:", tuple(selected_xyz.shape))
    print("  visible_preclose_xyz:", tuple(visible_preclose_xyz.shape))
    print("  visible_rel:", tuple(visible_rel.shape))
    print("  action:", tuple(action.shape))
    print("  waypoint_abs_max:", waypoint_abs_max)


if __name__ == "__main__":
    main()
