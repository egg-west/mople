import functools

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import FastGELUActivation, GELUActivation, NewGELUActivation, QuickGELUActivation

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def fuse_gelu(model):
    @torch.jit.script
    def gelu_fwd(x):
        return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

    @torch.jit.script
    def gelu_bwd(g, x):
        tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
        return ff * g

    class _FusedGeLUFunction(torch.autograd.Function):
        @staticmethod
        # bias is an optional argument
        def forward(ctx, input):
            ctx.input_tensor = input
            return gelu_fwd(input)

        @staticmethod
        def backward(ctx, grad_output):
            input = ctx.input_tensor
            tmp = gelu_bwd(grad_output, input)
            return tmp

    class FusedGelu(torch.nn.Module):
        def forward(self, input):
            return _FusedGeLUFunction.apply(input)

    fused_gelu_module = FusedGelu()
    hf_gelu_functions = [GELUActivation, FastGELUActivation, NewGELUActivation, QuickGELUActivation]

    for name, module in model.named_modules():
        for hf_gelu_function in hf_gelu_functions:
            if isinstance(module, hf_gelu_function):
                rsetattr(model, name, fused_gelu_module)

    return model

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean"):
        super().__init__(weight, size_average, ignore_index, reduce, "none")
        self._reduction = reduction

    def forward(self, input, target, mask=None):
        input = input.view(-1, input.size(-1))
        target = target.view(-1)

        if mask is not None:
            mask = mask.view(-1).bool()
            input = input[mask]
            target = target[mask]

        size = target.numel()

        loss = super().forward(input, target)

        if self._reduction == "none":
            return loss
        return loss.sum() / (size + 1e-8)


class PolyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean", epsilon=1.0):
        super().__init__()
        self.weight = torch.tensor(weight)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.cross_entropy = CrossEntropyLoss(weight, size_average, ignore_index, reduce, "none")
        self.epsilon = epsilon

    def forward(self, input, target, mask=None):
        if mask is not None:
            mask = mask.view(-1).bool()
            input = input.view(-1, input.size(-1))
            target = target.view(-1)
            input = input[mask]
            target = target[mask]

        onehot_target = F.one_hot(target, num_classes=input.size(-1)).to(device=input.device, dtype=input.dtype)
        pt = torch.sum(onehot_target * F.softmax(input, -1), -1)
        CE = self.cross_entropy(input, target)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


class RMLoss(nn.Module):
    def __init__(self, reduction="mean", beta=0.001):
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, logits, cu_lengths=None):
        # if cu_lengths is None, assume that all examples belong to the same conversation
        if cu_lengths is None:
            cu_lengths = [0, logits.size(0)]

        device = logits.device
        losses = []
        for start, end in zip(cu_lengths[:-1], cu_lengths[1:]):
            pairs = torch.combinations(torch.arange(end - start, device=device), 2)
            pos_ids, neg_ids = pairs[:, 0], pairs[:, 1]
            pos_logits = logits.take(start + pos_ids)
            neg_logits = logits.take(start + neg_ids)

            l2 = 0.5 * (pos_logits**2 + neg_logits**2)
            _loss = (-F.logsigmoid(pos_logits - neg_logits) + self.beta * l2).mean()
            losses.append(_loss)
        loss = torch.stack(losses)

        if self.reduction == "none":
            return loss
        return loss.mean()


class RMCLSLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean"):
        super().__init__(weight, size_average, ignore_index, reduce, "none")
        self._reduction = reduction

    def forward(self, logits, cu_lengths=None):
        # if cu_lengths is None, assume that all examples belong to the same conversation
        if cu_lengths is None:
            cu_lengths = [0, logits.size(0)]

        device = logits.device
        logit_pairs = []
        # aggregate combination between ranks
        for start, end in zip(cu_lengths[:-1], cu_lengths[1:]):
            pairs = torch.combinations(torch.arange(end - start, device=device), 2)
            pos_ids, neg_ids = pairs[:, 0], pairs[:, 1]
            pos_logits = logits.take(start + pos_ids)
            neg_logits = logits.take(start + neg_ids)
            merged = torch.stack((pos_logits, neg_logits), dim=1)
            logit_pairs.append(merged)
        logit_pairs = torch.concat(logit_pairs, dim=0)
        labels = torch.zeros(logit_pairs.shape[0], dtype=torch.long, device=device)
        loss = super().forward(logit_pairs, labels)

        if self._reduction == "none":
            return loss
        return loss.mean()

def get_loss(loss, poly_eps: float = 1.0, score_l2_reg: float = 0.001):
    if loss == "CrossEntropyLoss":
        return CrossEntropyLoss()
    elif loss == "Poly":
        return PolyLoss(epsilon=poly_eps)
    elif loss == "RMLoss":
        return RMLoss(beta=score_l2_reg)
    elif loss == "RMCLSLoss":
        return RMCLSLoss()
    else:
        raise ValueError(f"Loss {loss} not supported")
