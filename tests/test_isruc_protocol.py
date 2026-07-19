import numpy as np
import torch

import utils
from modeling_finetune import NeuralTransformer, resize_time_embedding


def _write_sample(root, subject=1, index=7):
    seq_dir = root / "seq" / f"ISRUC-group1-{subject}"
    label_dir = root / "labels" / f"ISRUC-group1-{subject}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    np.save(seq_dir / f"ISRUC-group1-{subject}-{index}.npy", np.zeros((20, 6, 6000), dtype=np.float32))
    np.save(label_dir / f"ISRUC-group1-{subject}-{index}.npy", np.arange(20, dtype=np.int64) % 5)


def test_isruc_loader_uses_numeric_pairing_and_contract(tmp_path):
    _write_sample(tmp_path, index=10)
    _write_sample(tmp_path, index=2)
    dataset = utils.ISRUCSequenceLoader(tmp_path, [1])
    signal, labels = dataset[0]
    assert tuple(signal.shape) == (20, 6, 30, 200)
    assert tuple(labels.shape) == (20,)
    assert dataset.samples[0][1] == 2
    assert dataset.get_ch_names() == ["F3", "C3", "O1", "F4", "C4", "O2"]


def test_isruc_temporal_resize_and_sequence_logits():
    assert tuple(resize_time_embedding(torch.zeros(1, 16, 200), 30).shape) == (1, 30, 200)
    model = NeuralTransformer(
        num_classes=5,
        depth=1,
        init_values=0,
        isruc_sequence=True,
        isruc_sequence_length=20,
    )
    with torch.no_grad():
        output = model(torch.zeros(1, 20, 6, 30, 200), input_chans=utils.get_input_chans(utils.ISRUC_LABRAM_CH))
    assert tuple(output.shape) == (1, 20, 5)
