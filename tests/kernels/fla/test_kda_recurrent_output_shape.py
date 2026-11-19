# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fused_recurrent_kda allocates the value shape when K and V differ."""

import torch

from vllm.third_party.flash_linear_attention.ops import kda as kda_ops


class _RecordingKernel:
    def __init__(self) -> None:
        self.output: torch.Tensor | None = None

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.output = kwargs["o"]

        return launch


def test_fused_recurrent_kda_output_matches_value_shape(monkeypatch):
    kernel = _RecordingKernel()
    monkeypatch.setattr(
        kda_ops, "fused_recurrent_gated_delta_rule_fwd_kernel", kernel
    )
    q = torch.zeros(1, 4, 1, 64, dtype=torch.float16)
    k = torch.ones_like(q)
    v = torch.ones(1, 4, 1, 32, dtype=torch.float16)
    out, _state = kda_ops.fused_recurrent_kda(
        q,
        k,
        v,
        -torch.ones_like(q, dtype=torch.float32),
        torch.ones(1, 4, 1, dtype=torch.float16),
        initial_state=torch.zeros(5, 1, 32, 64),
        cu_seqlens=torch.arange(5, dtype=torch.int32),
        ssm_state_indices=torch.arange(1, 5, dtype=torch.int32),
    )
    assert out.shape == v.shape
    assert out.dtype == k.dtype
    assert kernel.output is out
