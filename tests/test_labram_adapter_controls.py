import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling_finetune import NeuralTransformer
from optim_factory import get_parameter_groups


class _DummyAdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(3, 3)
        self.native_axis_adapter = torch.nn.Module()
        self.native_axis_adapter.patch_attn = torch.nn.Linear(3, 3)
        self.native_axis_adapter.alpha_patch = torch.nn.Parameter(torch.tensor(0.01))
        self.native_axis_adapter.alpha_depth = torch.nn.Parameter(torch.tensor(0.05))
        self.native_axis_adapter.depth_score = torch.nn.Linear(3, 1, bias=False)
        self.native_axis_adapter.depth_gate = torch.nn.Linear(3, 1, bias=False)


def test_alpha_parameters_get_a_separate_lr_group():
    model = _DummyAdapterModel()
    groups = get_parameter_groups(
        model,
        adapter_name_prefix="native_axis_adapter.",
        adapter_alpha_name_prefix="native_axis_adapter.alpha_",
        adapter_lr_scale=0.1,
        adapter_alpha_lr_scale=1.0,
        adapter_gate_name_prefix="native_axis_adapter.depth_gate.",
        adapter_gate_lr_scale=1.0,
    )
    assert any(
        group.get("is_adapter_alpha") and group["lr_scale"] == 1.0
        for group in groups
    )
    assert any(
        group.get("is_adapter") and not group.get("is_adapter_alpha")
        and group["lr_scale"] == 0.1
        for group in groups
    )
    assert any(
        any(parameter is model.native_axis_adapter.alpha_depth for parameter in group["params"])
        and group.get("is_adapter_alpha")
        and group["lr_scale"] == 1.0
        and group["weight_decay"] == 0.0
        for group in groups
    )
    assert any(
        any(parameter is model.native_axis_adapter.depth_score.weight for parameter in group["params"])
        and group.get("is_adapter") and not group.get("is_adapter_alpha")
        and group["lr_scale"] == 0.1
        for group in groups
    )
    assert any(
        any(parameter is model.native_axis_adapter.depth_gate.weight for parameter in group["params"])
        and group.get("is_adapter_gate")
        and group["lr_scale"] == 1.0
        and group["weight_decay"] == 0.0
        for group in groups
    )


def test_fixed_alpha_is_frozen_at_requested_value():
    model = NeuralTransformer(
        EEG_size=400,
        patch_size=200,
        in_chans=2,
        out_chans=8,
        num_classes=9,
        embed_dim=200,
        depth=2,
        num_heads=10,
        init_values=0.1,
        adapter_type="patch",
        adapter_fixed_alpha=0.01,
    )
    alpha = model.native_axis_adapter.alpha_patch
    assert abs(float(alpha.detach()) - 0.01) < 1e-6
    assert not alpha.requires_grad


def main():
    test_alpha_parameters_get_a_separate_lr_group()
    test_fixed_alpha_is_frozen_at_requested_value()
    print("adapter control tests: PASS")


if __name__ == "__main__":
    main()
