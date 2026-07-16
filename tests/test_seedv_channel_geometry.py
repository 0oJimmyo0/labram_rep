"""Verify that channel-only SEED-V adaptation uses the electrode axis."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling_finetune import NeuralTransformer


def test_channel_only_has_meaningful_channel_attention():
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
        adapter_type="channel",
        adapter_init_alpha=0.01,
        adapter_seed=12345,
    )
    samples = torch.randn(2, 62, 1, 200)
    model(samples).sum().backward()
    diagnostics = model.get_adapter_diagnostics()

    assert not hasattr(model.native_axis_adapter, "patch_attn")
    assert diagnostics["adapter_channel_count"] == 62
    assert diagnostics["adapter_patch_count"] == 1
    assert diagnostics["channel_attention_sequence_length"] == 62
    assert diagnostics["channel_q_grad_norm"] > 0.0
    assert diagnostics["channel_k_grad_norm"] > 0.0
    assert diagnostics["channel_v_grad_norm"] > 0.0
    assert diagnostics["channel_output_projection_grad_norm"] > 0.0
    assert diagnostics["raw_channel_ratio"] > 0.0


def main():
    test_channel_only_has_meaningful_channel_attention()
    print("SEED-V channel geometry: PASS")


if __name__ == "__main__":
    main()
