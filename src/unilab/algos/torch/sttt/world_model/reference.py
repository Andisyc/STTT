# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from pytorch/rl commit 1d94cfa5c35f9305b402fb71f0378f9cd7f09b28:
# torchrl/modules/models/model_based.py and
# sota-implementations/dreamer_v3/dreamer_v3_agent.py (_DreamerV3Decoder).
#
# MIT License
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Dependency-free Python DreamerV3 math; action scaling belongs to the caller.

The sole dynamics-math departure is removal of the reference block-GRU's
magnitude-dependent action clipping. The public model uses fixed reversible
per-joint scaling instead. No TensorDict, TorchRL, Triton or scan dependency.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SOURCE = {
    "repository": "https://github.com/pytorch/rl",
    "revision": "1d94cfa5c35f9305b402fb71f0378f9cd7f09b28",
    "rssm": "torchrl/modules/models/model_based.py",
    "vector_decoder": "sota-implementations/dreamer_v3/dreamer_v3_agent.py",
    "backend": "python",
    "action_scaling": "fixed_per_joint_division_no_dynamic_clipping",
}


def dreamer_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        std = 1.1368 / module.in_features**0.5
        nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(features))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(value.float(), (self.weight.shape[0],), self.weight.float(), self.eps).to(value.dtype)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int | None, hidden_dim: int, depth: int, eps: float):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend((nn.Linear(input_dim, hidden_dim), RMSNorm(hidden_dim, eps), nn.SiLU()))
            input_dim = hidden_dim
        if output_dim is not None:
            layers.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.apply(dreamer_init)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs[0] if len(inputs) == 1 else torch.cat(inputs, -1))


class BlockLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, blocks: int):
        super().__init__()
        if input_dim % blocks or output_dim % blocks:
            raise ValueError("block-linear input/output dimensions must divide evenly into blocks")
        self.blocks = blocks
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.empty(blocks, input_dim // blocks, output_dim // blocks))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        std = 1.1368 / input_dim**0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-2 * std, b=2 * std)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape[:-1]
        grouped = value.reshape(-1, self.blocks, self.input_dim // self.blocks).transpose(0, 1)
        result = torch.bmm(grouped, self.weight.to(value.dtype)).transpose(0, 1)
        return result.reshape(*shape, self.output_dim) + self.bias.to(value.dtype)


class BlockGRU(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, belief_dim: int, blocks: int, layers: int, eps: float):
        super().__init__()
        self.blocks = blocks
        self.belief_dim = belief_dim
        def projection(size: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(size, hidden_dim), RMSNorm(hidden_dim, eps), nn.SiLU())
        self.belief_projection = projection(belief_dim)
        self.state_projection = projection(state_dim)
        self.action_projection = projection(action_dim)
        width = belief_dim + 3 * hidden_dim * blocks
        hidden: list[nn.Module] = []
        for _ in range(layers):
            hidden.extend((BlockLinear(width, belief_dim, blocks), RMSNorm(belief_dim, eps), nn.SiLU()))
            width = belief_dim
        self.hidden_layers = nn.Sequential(*hidden)
        self.gates = BlockLinear(belief_dim, 3 * belief_dim, blocks)
        self.apply(dreamer_init)

    def forward(self, state: torch.Tensor, belief: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = torch.cat((self.belief_projection(belief), self.state_projection(state), self.action_projection(action)), -1)
        grouped = belief.reshape(*belief.shape[:-1], self.blocks, self.belief_dim // self.blocks)
        repeated = features.unsqueeze(-2).expand(*features.shape[:-1], self.blocks, features.shape[-1])
        hidden = self.hidden_layers(torch.cat((grouped, repeated), -1).flatten(-2))
        gates = self.gates(hidden).reshape(*hidden.shape[:-1], self.blocks, 3, self.belief_dim // self.blocks)
        reset, candidate, update = gates.unbind(-2)
        reset = reset.flatten(-2).sigmoid()
        candidate = (reset * candidate.flatten(-2)).tanh()
        update = (update.flatten(-2) - 1.0).sigmoid()
        return update * candidate + (1 - update) * belief


def categorical_probs(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    probs = logits.softmax(-1)
    return (1 - unimix) * probs + unimix / logits.shape[-1] if unimix else probs


def categorical_sample(logits: torch.Tensor, unimix: float) -> torch.Tensor:
    probs = categorical_probs(logits, unimix)
    uniform = torch.rand(probs.shape[:-1], device=probs.device)
    indices = (uniform.unsqueeze(-1) > probs.cumsum(-1)).sum(-1).clamp_max(probs.shape[-1] - 1)
    hard = torch.zeros_like(probs).scatter_(-1, indices.unsqueeze(-1), 1.0)
    return probs + (hard - probs).detach()


class RSSMPrior(nn.Module):
    def __init__(self, *, state_dim: int, action_dim: int, hidden_dim: int, belief_dim: int, groups: int, classes: int, recurrent_model: str, blocks: int, layers: int, prior_layers: int, eps: float):
        super().__init__()
        self.groups, self.classes = groups, classes
        self.recurrent_model = recurrent_model
        if recurrent_model == "block_gru":
            self.rnn = BlockGRU(state_dim, action_dim, hidden_dim, belief_dim, blocks, layers, eps)
            self.projector = None
            self.predictor = MLP(belief_dim, state_dim, hidden_dim, prior_layers, eps)
        else:
            self.projector = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.SiLU())
            self.rnn = nn.GRUCell(hidden_dim, belief_dim)
            self.predictor = nn.Sequential(nn.Linear(belief_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, state_dim))

    def forward(self, state: torch.Tensor, belief: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = state.flatten(-2)
        if self.recurrent_model == "block_gru":
            belief = self.rnn(state, belief, action)
        else:
            projected = self.projector(torch.cat((state, action), -1))
            with torch.autocast(device_type=projected.device.type, enabled=False):
                compute_dtype = torch.float64 if self.rnn.weight_hh.dtype == torch.float64 else torch.float32
                belief = self.rnn(projected.to(compute_dtype), belief.to(compute_dtype)).to(projected.dtype)
        logits = self.predictor(belief).reshape(*belief.shape[:-1], self.groups, self.classes)
        return belief, logits


class RSSMPosterior(nn.Module):
    def __init__(self, belief_dim: int, embed_dim: int, hidden_dim: int, groups: int, classes: int, layers: int, eps: float):
        super().__init__()
        self.groups, self.classes = groups, classes
        self.predictor = MLP(belief_dim + embed_dim, groups * classes, hidden_dim, layers, eps)

    def forward(self, belief: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        return self.predictor(belief, embedding).reshape(*belief.shape[:-1], self.groups, self.classes)


class VectorDecoder(nn.Module):
    """Reference shared normalized trunk, independent event output heads."""
    def __init__(self, input_dim: int, event_dims: tuple[int, ...], hidden_dim: int, depth: int, eps: float):
        super().__init__()
        self.backbone = MLP(input_dim, None, hidden_dim, depth, eps)
        self.output_heads = nn.ModuleList(nn.Linear(hidden_dim, size) for size in event_dims)
        self.output_heads.apply(dreamer_init)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(features)
        return torch.cat(tuple(head(hidden) for head in self.output_heads), -1)
