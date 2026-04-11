import torch
import torch.nn as nn
import math

class NestedLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, max_rank=16, alpha=16.0, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.alpha = alpha
        self.active_rank = max_rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.lora_A = nn.Parameter(torch.empty(in_features, max_rank))
        self.lora_B = nn.Parameter(torch.zeros(max_rank, out_features))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def scaling(self):
        return self.alpha / self.active_rank if self.active_rank > 0 else 0.0

    def forward(self, x):
        base = nn.functional.linear(x, self.weight, self.bias_param)

        if self.active_rank > 0:
            A = self.lora_A[:, :self.active_rank]
            B = self.lora_B[:self.active_rank, :]
            return base + (x @ A @ B) * self.scaling

        return base


def inject_nested_lora(model, target_modules, max_rank=16, alpha=16.0):
    adapters = {}

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if not any(t in name for t in target_modules):
            continue

        new_layer = NestedLoRALinear(
            module.in_features,
            module.out_features,
            max_rank=max_rank,
            alpha=alpha,
            bias=module.bias is not None,
        )

        new_layer.weight.data.copy_(module.weight.data)

        if module.bias is not None:
            new_layer.bias_param.data.copy_(module.bias.data)

        parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr_name, new_layer)

        adapters[name] = new_layer

    return adapters


def set_rank(model, rank):
    for m in model.modules():
        if isinstance(m, NestedLoRALinear):
            m.active_rank = min(rank, m.max_rank)
