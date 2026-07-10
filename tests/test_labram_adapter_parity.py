"""Deterministic dense-vs-adapter parity check for the LaBraM-native adapter."""

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling_finetune import NeuralTransformer


def build(adapter_type):
    return NeuralTransformer(
        EEG_size=400,
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
        num_classes=3,
        adapter_type=adapter_type,
        adapter_gamma=0.0,
        adapter_gamma_zero_skip_branch=True,
        adapter_seed=12345,
    )


def main():
    torch.manual_seed(7)
    dense = build("none").eval()
    torch.manual_seed(7)
    adapted = build("channel_patch").eval()

    dense_state = dense.state_dict()
    missing, unexpected = adapted.load_state_dict(dense_state, strict=False)
    adapter_only = [key for key in missing if key.startswith("native_axis_adapter.")]
    if unexpected or len(adapter_only) == 0:
        raise AssertionError(f"Unexpected state-dict mismatch: missing={missing}, unexpected={unexpected}")

    samples = torch.randn(2, 32, 2, 200)
    with torch.no_grad():
        dense_features = dense.forward_features(samples)
        adapted_features = adapted.forward_features(samples)
        dense_logits = dense(samples)
        adapted_logits = adapted(samples)

    feature_diff = (dense_features - adapted_features).abs().max().item()
    logit_diff = (dense_logits - adapted_logits).abs().max().item()
    print(f"max_feature_diff={feature_diff:.3e}")
    print(f"max_logit_diff={logit_diff:.3e}")
    if feature_diff > 1e-6 or logit_diff > 1e-6:
        raise AssertionError("gamma-zero adapter path does not match dense LaBraM")


if __name__ == "__main__":
    main()
