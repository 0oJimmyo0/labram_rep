"""Run the channel-axis mechanism check on one real SEED-V batch."""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils
from modeling_finetune import labram_base_patch200_200


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = dict(checkpoint.get("model", checkpoint))
    state.pop("head.weight", None)
    state.pop("head.bias", None)
    model.load_state_dict(state, strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="/data/neurogroup/mingyangjiang/data/SEED-V_processed_lmdb")
    parser.add_argument("--manifest", default="docs/seedv_channel_manifest_provisional.json")
    parser.add_argument("--checkpoint", default="checkpoints/labram-base.pth")
    args = parser.parse_args()

    torch.set_num_threads(2)
    dataset = utils.SEEDVLoader(args.data_path, mode="train", channel_manifest=args.manifest)
    samples = torch.stack([dataset[index][0] for index in (0, 1)]).float() / 100.0
    targets = torch.tensor([dataset[index][1] for index in (0, 1)], dtype=torch.long)
    input_chans = utils.get_input_chans(dataset.get_ch_names())

    model = labram_base_patch200_200(
        pretrained=False,
        num_classes=5,
        drop_path_rate=0.1,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
        adapter_type="channel",
        adapter_num_heads=4,
        adapter_init_alpha=0.01,
        adapter_seed=12345,
    ).eval()
    load_checkpoint(model, args.checkpoint)

    logits = model(samples, input_chans=input_chans)
    torch.nn.functional.cross_entropy(logits, targets).backward()
    diagnostics = model.get_adapter_diagnostics()

    print(f"sample_shape={tuple(samples.shape)}")
    print(f"channel_attention_sequence_length={diagnostics['channel_attention_sequence_length']:.6e}")
    for name in (
        "channel_q_grad_norm",
        "channel_k_grad_norm",
        "channel_v_grad_norm",
        "channel_output_projection_grad_norm",
        "raw_channel_ratio",
    ):
        print(f"{name}={diagnostics[name]:.6e}")

    assert tuple(samples.shape) == (2, 62, 1, 200)
    assert diagnostics["channel_attention_sequence_length"] == 62
    assert diagnostics["channel_q_grad_norm"] > 0.0
    assert diagnostics["channel_k_grad_norm"] > 0.0
    assert diagnostics["channel_v_grad_norm"] > 0.0
    assert diagnostics["channel_output_projection_grad_norm"] > 0.0
    assert diagnostics["raw_channel_ratio"] > 0.0
    print("SEED-V real channel geometry: PASS")


if __name__ == "__main__":
    main()
