"""Verify that SEED-V singleton patches are not reported as temporal mixing."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling_finetune import NeuralTransformer


def test_seedv_singleton_patch_attention_geometry_and_gradients():
    torch.manual_seed(13)
    model = NeuralTransformer(
        EEG_size=200,
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
        adapter_type="patch",
        adapter_num_heads=4,
        adapter_seed=12345,
    )
    model.eval()

    samples = torch.randn(2, 62, 1, 200)
    logits = model(samples)
    logits.sum().backward()
    diagnostics = model.get_adapter_diagnostics()

    assert tuple(samples.shape[1:]) == (62, 1, 200)
    assert diagnostics["adapter_channel_count"] == 62
    assert diagnostics["adapter_patch_count"] == 1
    assert diagnostics["patch_attention_sequence_length"] == 1
    assert diagnostics["patch_temporal_interactions_active"] == 0
    assert diagnostics["patch_q_grad_norm"] <= 1e-8
    assert diagnostics["patch_k_grad_norm"] <= 1e-8
    assert diagnostics["patch_v_grad_norm"] > 0.0
    assert diagnostics["patch_output_projection_grad_norm"] > 0.0


def main():
    test_seedv_singleton_patch_attention_geometry_and_gradients()
    print("SEED-V singleton patch geometry: PASS")


if __name__ == "__main__":
    main()
