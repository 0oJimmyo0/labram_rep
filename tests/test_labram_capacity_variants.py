"""Controls for explicit singleton-patch capacity variants."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling_finetune import NeuralTransformer
from optim_factory import get_parameter_groups


def build(variant, patch_output_dropout=0.0):
    return NeuralTransformer(
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
        adapter_variant=variant,
        adapter_patch_output_dropout=patch_output_dropout,
        adapter_bottleneck=64,
        adapter_num_heads=4,
        adapter_init_alpha=0.01,
        adapter_seed=12345,
    )


def test_variants_are_explicit_and_parameterized():
    full = build("full")
    dropped = build("output_dropout", 0.1)
    bottleneck = build("bottleneck")

    assert hasattr(full.native_axis_adapter, "patch_attn")
    assert hasattr(dropped.native_axis_adapter, "patch_attn")
    assert isinstance(dropped.native_axis_adapter.patch_output_dropout, torch.nn.Dropout)
    assert dropped.native_axis_adapter.patch_output_dropout.p == 0.1
    dropped.eval()
    dropped_input = torch.randn(2, 62, 1, 200)
    with torch.no_grad():
        dropped_eval_a = dropped(dropped_input)
        dropped_eval_b = dropped(dropped_input)
    assert torch.equal(dropped_eval_a, dropped_eval_b)
    assert not hasattr(bottleneck.native_axis_adapter, "patch_attn")
    assert hasattr(bottleneck.native_axis_adapter, "singleton_patch_residual")
    assert sum(p.numel() for p in full.native_axis_adapter.parameters()) == 161201
    assert sum(p.numel() for p in dropped.native_axis_adapter.parameters()) == 161201
    assert sum(p.numel() for p in bottleneck.native_axis_adapter.parameters()) == 26265


def test_gamma_zero_parity_and_variant_gradients():
    samples = torch.randn(2, 62, 1, 200)
    for variant, dropout in (("output_dropout", 0.1), ("bottleneck", 0.0)):
        torch.manual_seed(7)
        dense = NeuralTransformer(
            EEG_size=200, patch_size=200, in_chans=1, out_chans=8,
            embed_dim=200, depth=2, num_heads=4, mlp_ratio=2,
            qkv_bias=False, use_abs_pos_emb=False, use_rel_pos_bias=False,
            use_mean_pooling=True, init_values=0.1, num_classes=5,
            adapter_type="none", adapter_gamma=0.0,
        ).eval()
        torch.manual_seed(7)
        adapted = build(variant, dropout)
        adapted.adapter_gamma = 0.0
        adapted.eval()
        adapted.load_state_dict(dense.state_dict(), strict=False)
        with torch.no_grad():
            assert torch.allclose(dense(samples), adapted(samples), atol=1e-6, rtol=0.0)

        adapted.adapter_gamma = 1.0
        adapted.train()
        adapted(samples).sum().backward()
        diagnostics = adapted.get_adapter_diagnostics()
        assert diagnostics["adapter_core_grad_norm"] > 0.0
        assert diagnostics["alpha_grad_norm"] > 0.0


def test_variant_optimizer_groups_use_core_and_alpha_rates():
    model = build("bottleneck")
    groups = get_parameter_groups(
        model,
        adapter_name_prefix="native_axis_adapter.",
        adapter_alpha_name_prefix="native_axis_adapter.alpha_",
        adapter_lr_scale=1.0,
        adapter_alpha_lr_scale=0.1,
    )
    assert any(group.get("is_adapter") and not group.get("is_adapter_alpha")
               and group["lr_scale"] == 1.0 for group in groups)
    assert any(group.get("is_adapter_alpha") and group["lr_scale"] == 0.1
               for group in groups)


def main():
    test_variants_are_explicit_and_parameterized()
    test_gamma_zero_parity_and_variant_gradients()
    test_variant_optimizer_groups_use_core_and_alpha_rates()
    print("capacity variant controls: PASS")


if __name__ == "__main__":
    main()
