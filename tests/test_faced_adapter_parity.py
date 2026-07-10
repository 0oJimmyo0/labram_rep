"""Dense-vs-adapter identity check on real FACED samples and checkpoint weights."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
from modeling_finetune import labram_base_patch200_200


def load_checkpoint_state(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = dict(checkpoint.get("model", checkpoint))
    for key in ("head.weight", "head.bias"):
        state.pop(key, None)
    return state


def build_model(adapter_type="none", gamma=1.0, gamma_zero_skip=False):
    return labram_base_patch200_200(
        pretrained=False,
        num_classes=9,
        drop_path_rate=0.1,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
        adapter_type=adapter_type,
        adapter_gamma=gamma,
        adapter_gamma_zero_skip_branch=gamma_zero_skip,
        adapter_seed=12345,
    ).eval()


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="/data/neurogroup/mingyangjiang/data/FACED")
    parser.add_argument("--checkpoint", default="./checkpoints/labram-base.pth")
    args = parser.parse_args()

    train_dataset, _, _ = utils.prepare_FACED_dataset(args.data_path)
    input_chans = utils.get_input_chans(train_dataset.get_ch_names())
    samples = torch.stack([train_dataset[index][0] for index in (0, 1)]).float() / 100

    torch.manual_seed(17)
    dense = build_model()
    torch.manual_seed(17)
    adapted = build_model("channel_patch", gamma=0.0, gamma_zero_skip=True)
    state = load_checkpoint_state(args.checkpoint)
    dense.load_state_dict(state, strict=False)
    adapted.load_state_dict(state, strict=False)

    with torch.no_grad():
        dense_features = dense.forward_features(samples, input_chans=input_chans)
        adapted_features = adapted.forward_features(samples, input_chans=input_chans)
        dense_logits = dense(samples, input_chans=input_chans)
        adapted_logits = adapted(samples, input_chans=input_chans)

    feature_diff = (dense_features - adapted_features).abs().max().item()
    logit_diff = (dense_logits - adapted_logits).abs().max().item()
    print(f"sample_shape={tuple(samples.shape)} input_chans={len(input_chans)}")
    print(f"max_abs_feature_difference={feature_diff:.3e}")
    print(f"max_abs_logit_difference={logit_diff:.3e}")
    if feature_diff > 1e-6 or logit_diff > 1e-6:
        raise AssertionError("gamma-zero FACED path does not match dense LaBraM")


if __name__ == "__main__":
    main()
