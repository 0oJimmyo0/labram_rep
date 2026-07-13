"""Checks for the LaBraM-native upper-depth AttnRes-inspired adapter mode."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling_finetune import NeuralTransformer


def build(depth_mode="none"):
    return NeuralTransformer(
        EEG_size=400,
        patch_size=200,
        in_chans=1,
        out_chans=8,
        embed_dim=200,
        depth=4,
        num_heads=4,
        mlp_ratio=2,
        qkv_bias=False,
        use_abs_pos_emb=False,
        use_rel_pos_bias=False,
        use_mean_pooling=True,
        init_values=0.1,
        num_classes=3,
        adapter_type="patch",
        adapter_init_alpha=0.01,
        adapter_depth_mode=depth_mode,
        adapter_depth_k=2,
        adapter_seed=12345,
    )


def test_zero_depth_mix_matches_patch_only():
    torch.manual_seed(19)
    patch_only = build("none").eval()
    torch.manual_seed(19)
    depth_model = build("lastk_attnres").eval()

    missing, unexpected = depth_model.load_state_dict(patch_only.state_dict(), strict=False)
    assert not unexpected
    assert all(key.startswith("native_axis_adapter.depth_") for key in missing)

    samples = torch.randn(2, 32, 2, 200)
    with torch.no_grad():
        patch_logits = patch_only(samples)
        depth_logits = depth_model(samples)
    assert (patch_logits - depth_logits).abs().max().item() <= 1e-6


def test_depth_weights_normalize_and_receive_gradient():
    torch.manual_seed(23)
    model = build("lastk_attnres")
    model.native_axis_adapter.depth_mix.data.fill_(0.5)
    model.native_axis_adapter.depth_query.data[0] = 1.0

    samples = torch.randn(2, 32, 2, 200)
    logits = model(samples)
    logits.square().mean().backward()

    adapter = model.native_axis_adapter
    weights = adapter._last_depth_weights
    assert weights is not None
    assert weights.shape == (2,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.std(weights) > 0
    assert adapter.depth_query.grad is not None
    assert adapter.depth_query.grad.abs().sum() > 0
    assert adapter.depth_mix.grad is not None
    assert adapter.depth_mix.grad.abs().sum() > 0


def test_legacy_delta_mode_still_forwards():
    torch.manual_seed(29)
    model = build("lastk_delta").eval()
    samples = torch.randn(2, 32, 2, 200)
    with torch.no_grad():
        logits = model(samples)
    assert logits.shape == (2, 3)


def main():
    test_zero_depth_mix_matches_patch_only()
    test_depth_weights_normalize_and_receive_gradient()
    print("depth adapter tests: PASS")


if __name__ == "__main__":
    main()
