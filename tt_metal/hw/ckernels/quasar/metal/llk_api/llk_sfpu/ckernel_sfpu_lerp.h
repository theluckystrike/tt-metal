// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_binary.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_lerp(
    const uint dst_index_in0,  // input (start)
    const uint dst_index_in1,  // end
    const uint dst_index_in2,  // weight
    const uint dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_lerp(). Supported data formats are: Float32, Float16_b.");

    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    // lerp: out = input + weight * (end - input), anchored on the nearer endpoint:
    //   weight <  0.5: input + weight * (end - input)
    //   weight >= 0.5: end + (weight - 1) * (end - input)
    // Anchoring on end for the upper half keeps the low bits of end that end - input rounds
    // away when |input| >> |end|, and makes lerp(input, end, 1) == end exactly. torch.lerp
    // switches on |weight| >= 0.5; negative weights stay on input here because SFPMAD's
    // product overflows before the add, and |weight - 1| > |weight| would overflow earlier.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat in2 = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];
        sfpi::vFloat diff = in1 - in0;
        v_if(in2 >= 0.5f) {
            in2 = in2 - 1.0f;
            in0 = in1;
        }
        v_endif;
        sfpi::vFloat result = in0 + in2 * diff;
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
