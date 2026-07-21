"""Tests for the explicit frozen-backbone SEED-V matrix contract."""

from pathlib import Path
import sys
import gc

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling_finetune import NeuralTransformer
from run_class_finetuning import (
    assert_optimizer_matches_trainability,
    configure_labram_trainability,
)


def build_model(adapter_type="none"):
    return NeuralTransformer(
        EEG_size=200,
        patch_size=200,
        in_chans=1,
        out_chans=8,
        embed_dim=200,
        depth=1,
        num_heads=4,
        mlp_ratio=2,
        qkv_bias=False,
        use_abs_pos_emb=False,
        use_rel_pos_bias=False,
        use_mean_pooling=True,
        init_values=0.1,
        num_classes=5,
        adapter_type=adapter_type,
        adapter_num_heads=4,
        adapter_init_alpha=0.01,
        adapter_seed=12345,
    )


def test_seedv_frozen_trainability_and_optimizer_membership():
    for adapter_type in ("none", "channel", "patch"):
        model = build_model(adapter_type)
        summary = configure_labram_trainability(model, "frozen")
        trainable = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        assert summary["backbone"]["trainable"] == 0
        assert summary["head"]["trainable"] > 0
        if adapter_type == "none":
            assert not any(name.startswith("native_axis_adapter.") for name in trainable)
        else:
            assert any(name.startswith("native_axis_adapter.") for name in trainable)

        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        assert_optimizer_matches_trainability(model, optimizer)
        del optimizer, model
        gc.collect()


def test_seedv_channel_only_wiring():
    model = build_model("channel")
    adapter = model.native_axis_adapter
    assert hasattr(adapter, "channel_attn")
    assert hasattr(adapter, "alpha_channel")
    assert not hasattr(adapter, "patch_attn")
    assert not hasattr(adapter, "singleton_patch_residual")
    assert not hasattr(adapter, "alpha_patch")

    logits = model(torch.randn(2, 62, 1, 200))
    logits.square().mean().backward()
    diagnostics = model.get_adapter_diagnostics()
    assert logits.shape == (2, 5)
    assert diagnostics["adapter_channel_count"] == 62
    assert diagnostics["adapter_patch_count"] == 1
    assert diagnostics["channel_q_grad_norm"] > 0.0
    assert diagnostics["channel_k_grad_norm"] > 0.0
    assert diagnostics["channel_v_grad_norm"] > 0.0
    assert "raw_patch_ratio" not in diagnostics
    del logits, model
    gc.collect()


def test_seedv_patch_is_singleton_capacity_control():
    model = build_model("patch")
    logits = model(torch.randn(2, 62, 1, 200))
    logits.square().mean().backward()
    diagnostics = model.get_adapter_diagnostics()
    assert logits.shape == (2, 5)
    assert diagnostics["patch_attention_sequence_length"] == 1
    assert diagnostics["patch_temporal_interactions_active"] == 0
    assert diagnostics["patch_q_grad_norm"] <= 1e-8
    assert diagnostics["patch_k_grad_norm"] <= 1e-8
    assert "raw_channel_ratio" not in diagnostics
    del logits, model
    gc.collect()


def main():
    test_seedv_frozen_trainability_and_optimizer_membership()
    test_seedv_channel_only_wiring()
    test_seedv_patch_is_singleton_capacity_control()
    print("SEED-V frozen axis matrix: PASS")


if __name__ == "__main__":
    main()
