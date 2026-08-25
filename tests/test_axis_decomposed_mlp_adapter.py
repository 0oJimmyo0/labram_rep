"""Checks for the operator-matched, non-mixing native-axis control."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling_finetune import NeuralTransformer


def build(zero_init_output=False):
    return NeuralTransformer(
        EEG_size=1000,
        patch_size=200,
        in_chans=1,
        out_chans=8,
        embed_dim=200,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        qkv_bias=False,
        use_abs_pos_emb=False,
        use_rel_pos_bias=False,
        use_mean_pooling=True,
        init_values=0.1,
        num_classes=5,
        adapter_type="channel_patch",
        adapter_variant="low_rank",
        adapter_operator="mlp",
        adapter_bottleneck=106,
        adapter_num_heads=4,
        adapter_dropout=0.0,
        adapter_init_alpha=0.01,
        adapter_gamma=1.0,
        adapter_zero_init_output=zero_init_output,
        adapter_seed=12345,
    )


def test_axis_decomposed_mlp_geometry_capacity_and_gradients():
    model = build().train()
    adapter = model.native_axis_adapter
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 86214
    assert adapter.adapter_operator == "mlp"
    assert not hasattr(adapter, "channel_attn")
    assert not hasattr(adapter, "patch_attn")
    assert hasattr(adapter, "channel_down")
    assert hasattr(adapter, "patch_down")

    output = model.forward_features(torch.randn(2, 6, 5, 200), return_all_tokens=True)
    assert output.shape == (2, 31, 200)
    output.sum().backward()
    assert adapter.channel_down.weight.grad is not None
    assert adapter.patch_up.weight.grad is not None
    diagnostics = model.get_adapter_diagnostics()
    assert diagnostics["channel_spatial_interactions_active"] == 0
    assert diagnostics["patch_temporal_interactions_active"] == 0
    assert diagnostics["channel_axis_branch_active"] == 1
    assert diagnostics["patch_axis_branch_active"] == 1


def test_axis_decomposed_mlp_zero_initializes_residual_output():
    model = build(zero_init_output=True).eval()
    with torch.no_grad():
        output = model.forward_features(torch.randn(2, 6, 5, 200), return_all_tokens=True)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(model.native_axis_adapter.channel_up.weight) == 0
    assert torch.count_nonzero(model.native_axis_adapter.patch_up.weight) == 0

