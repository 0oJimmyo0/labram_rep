"""Parity and geometry checks for the common low-rank native-axis adapter."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling_finetune import NeuralTransformer


def build(adapter_type="channel_patch", gamma=1.0):
    return NeuralTransformer(
        EEG_size=6000,
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
        adapter_type=adapter_type,
        adapter_variant="low_rank",
        adapter_bottleneck=32,
        adapter_num_heads=4,
        adapter_init_alpha=0.01,
        adapter_gamma=gamma,
        adapter_seed=12345,
    )


def test_low_rank_geometry_and_parameter_reduction():
    model = build().eval()
    adapter = model.native_axis_adapter
    assert hasattr(adapter, "channel_down")
    assert hasattr(adapter, "channel_up")
    assert hasattr(adapter, "patch_down")
    assert hasattr(adapter, "patch_up")
    assert adapter.channel_attn.embed_dim == 32
    assert adapter.patch_attn.embed_dim == 32
    assert sum(p.numel() for p in adapter.parameters()) < 161201 * 2

    with torch.no_grad():
        model(torch.randn(2, 6, 30, 200))
    diagnostics = model.get_adapter_diagnostics()
    assert diagnostics["adapter_channel_count"] == 6
    assert diagnostics["adapter_patch_count"] == 30
    assert diagnostics["channel_spatial_interactions_active"] == 1
    assert diagnostics["patch_temporal_interactions_active"] == 1


def test_low_rank_single_axis_variants_construct():
    for adapter_type in ("channel", "patch", "channel_patch"):
        model = build(adapter_type=adapter_type)
        adapter = model.native_axis_adapter
        if adapter_type in {"channel", "channel_patch"}:
            assert hasattr(adapter, "channel_down")
            assert hasattr(adapter, "channel_up")
        else:
            assert not hasattr(adapter, "channel_down")
        if adapter_type in {"patch", "channel_patch"}:
            assert hasattr(adapter, "patch_down")
            assert hasattr(adapter, "patch_up")
        else:
            assert not hasattr(adapter, "patch_down")


def test_low_rank_gamma_zero_parity_and_gradients():
    samples = torch.randn(2, 6, 30, 200)
    torch.manual_seed(7)
    dense = NeuralTransformer(
        EEG_size=6000, patch_size=200, in_chans=1, out_chans=8,
        embed_dim=200, depth=2, num_heads=4, mlp_ratio=2,
        qkv_bias=False, use_abs_pos_emb=False, use_rel_pos_bias=False,
        use_mean_pooling=True, init_values=0.1, num_classes=5,
        adapter_type="none",
    ).eval()
    torch.manual_seed(7)
    adapted = build(gamma=0.0).eval()
    adapted.load_state_dict(dense.state_dict(), strict=False)
    with torch.no_grad():
        assert torch.allclose(dense(samples), adapted(samples), atol=1e-6, rtol=0.0)

    adapted.adapter_gamma = 1.0
    adapted.train()
    adapted(samples).sum().backward()
    diagnostics = adapted.get_adapter_diagnostics()
    assert diagnostics["adapter_core_grad_norm"] > 0.0
    assert diagnostics["alpha_grad_norm"] > 0.0


if __name__ == "__main__":
    test_low_rank_geometry_and_parameter_reduction()
    test_low_rank_single_axis_variants_construct()
    test_low_rank_gamma_zero_parity_and_gradients()
    print("low-rank adapter tests: PASS")
