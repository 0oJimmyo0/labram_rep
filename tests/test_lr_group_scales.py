import torch
from torch import nn

from optim_factory import get_parameter_groups


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)
        self.native_axis_adapter = nn.Module()
        self.native_axis_adapter.alpha_channel = nn.Parameter(torch.tensor(0.01))
        self.native_axis_adapter.core = nn.Linear(4, 4)
        self.head = nn.Linear(4, 3)


def _group_for(groups, parameter):
    parameter_id = id(parameter)
    return next(group for group in groups if any(id(item) == parameter_id for item in group['params']))


def test_backbone_head_and_adapter_lr_precedence():
    model = TinyModel()
    groups = get_parameter_groups(
        model,
        weight_decay=0.03,
        get_num_layer=lambda _name: 0,
        get_layer_scale=lambda _layer: 1.0,
        adapter_name_prefix='native_axis_adapter.',
        adapter_alpha_name_prefix='native_axis_adapter.alpha_',
        adapter_lr_scale=1.0,
        adapter_alpha_lr_scale=0.1,
        backbone_lr_scale=0.1,
        head_lr_scale=1.0,
    )

    assert _group_for(groups, model.blocks[0].weight)['lr_scale'] == 0.1
    assert _group_for(groups, model.native_axis_adapter.core.weight)['lr_scale'] == 1.0
    assert _group_for(groups, model.native_axis_adapter.alpha_channel)['lr_scale'] == 0.1
    assert _group_for(groups, model.head.weight)['lr_scale'] == 1.0

    assert _group_for(groups, model.blocks[0].weight)['is_backbone']
    assert _group_for(groups, model.head.weight)['is_head']
    assert _group_for(groups, model.native_axis_adapter.core.weight)['is_adapter']


if __name__ == '__main__':
    test_backbone_head_and_adapter_lr_precedence()
