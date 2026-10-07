# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.lerp for weight >= 0.5, where start + weight * (end - start) loses end.

When |start| is much larger than |end|, end - start rounds away the low bits of end, and the
one-sided form adds start back without ever recovering them: lerp(1000, 0.1, 1) comes out as
0.099975586 instead of 0.1, and lerp(1e8, 1, 1) as 0. torch.lerp switches to
end - (end - start) * (1 - weight) for |weight| >= 0.5, which is exact at weight == 1.

Start and end share a sign here, so the result lies between them and there is no cancellation
in the exact answer: every ULP of error comes from the formula. The hardware-free reproduction
of the kernel arithmetic is lerp_sfpu_numpy_port.py next to this file.
"""

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_ulp

pytestmark = pytest.mark.use_module_device

# (dtype, start range, end range): bf16 needs a wider magnitude gap before end - start rounds.
DOMAINS = {
    "float32": (torch.float32, (100.0, 1.0e4), (0.01, 1.0)),
    "bfloat16": (torch.bfloat16, (1.0e4, 1.0e6), (1.0e-3, 1.0)),
}


def make_inputs(dtype_name, shape):
    torch_dtype, (start_lo, start_hi), (end_lo, end_hi) = DOMAINS[dtype_name]
    gen = torch.Generator().manual_seed(0)
    start = torch.empty(shape, dtype=torch.float32).uniform_(start_lo, start_hi, generator=gen).to(torch_dtype)
    end = torch.empty(shape, dtype=torch.float32).uniform_(end_lo, end_hi, generator=gen).to(torch_dtype)
    return start, end, gen


@pytest.mark.parametrize("shape", [(64, 128)])
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("weight", [1.0, 0.9375, 0.75, 0.5, 0.25])
def test_lerp_upper_weight_scalar(device, shape, dtype_name, weight):
    torch_start, torch_end, _ = make_inputs(dtype_name, shape)
    golden = ttnn.get_golden_function(ttnn.lerp)(torch_start, torch_end, weight)

    start = ttnn.from_torch(torch_start, layout=ttnn.TILE_LAYOUT, device=device)
    end = ttnn.from_torch(torch_end, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.lerp(start, end, weight))

    if weight == 1.0:
        assert torch.equal(result, torch_end), "lerp(start, end, 1) must return end exactly"
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


@pytest.mark.parametrize("shape", [(64, 128)])
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("weight_range", [(0.5, 1.0), (0.0, 0.5), (-1.0, 0.0)])
def test_lerp_upper_weight_tensor(device, shape, dtype_name, weight_range):
    torch_start, torch_end, gen = make_inputs(dtype_name, shape)
    torch_weight = torch.empty(shape, dtype=torch.float32).uniform_(*weight_range, generator=gen).to(torch_start.dtype)
    golden = ttnn.get_golden_function(ttnn.lerp)(torch_start, torch_end, torch_weight)

    start = ttnn.from_torch(torch_start, layout=ttnn.TILE_LAYOUT, device=device)
    end = ttnn.from_torch(torch_end, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.lerp(start, end, weight))

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)
