"""Geometry and trainability contract for the LaBraM PhysioNet-MI adapter."""

import torch

import utils
from modeling_finetune import labram_base_patch200_200


def build(adapter_type):
    return labram_base_patch200_200(
        num_classes=4,
        qkv_bias=False,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        init_values=0.1,
        adapter_type=adapter_type,
        adapter_variant="low_rank",
        adapter_bottleneck=64,
        adapter_num_heads=4,
        adapter_init_alpha=0.01,
    )


def test_physio_channel_patch_geometry_is_active():
    model = build("channel_patch").eval()
    samples = torch.randn(1, 64, 4, 200)
    with torch.no_grad():
        model.forward_features(
            samples,
            input_chans=utils.get_input_chans(utils.PHYSIONET_MI_LABRAM_CH),
        )
    diagnostics = model.get_adapter_diagnostics()
    assert diagnostics["adapter_channel_count"] == 64
    assert diagnostics["adapter_patch_count"] == 4
    assert diagnostics["channel_spatial_interactions_active"] == 1
    assert diagnostics["patch_temporal_interactions_active"] == 1


def test_physio_native_axes_have_separate_low_rank_parameters():
    model = build("channel_patch")
    names = dict(model.native_axis_adapter.named_parameters())
    assert "channel_down.weight" in names
    assert "channel_up.weight" in names
    assert "patch_down.weight" in names
    assert "patch_up.weight" in names
    assert names["channel_down.weight"].shape == (64, 200)
    assert names["patch_down.weight"].shape == (64, 200)
    assert names["channel_attn.in_proj_weight"].shape == (192, 64)
    assert names["patch_attn.in_proj_weight"].shape == (192, 64)
