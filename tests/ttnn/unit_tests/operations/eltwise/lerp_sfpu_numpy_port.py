# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Hardware-free reproduction of the ttnn.lerp SFPU accuracy loss for weight >= 0.5.

Ports calculate_lerp (tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_lerp.h)
line for line to float32 numpy (sfpi::vFloat is fp32, SFPMAD is a fused multiply-add) and compares it
against the exactly rounded result of start + weight * (end - start), computed with Python fractions.

    old kernel:  result = in0 + in2 * (in1 - in0)
    new kernel:  result = coeff * (in1 - in0) + base, with
                 coeff, base = (in2 - 1, in1) if in2 >= 0.5 else (in2, in0)

This is torch.lerp's two-sided form (aten lerp_vec), so it also returns end exactly at weight == 1.
torch switches on |weight| >= 0.5; the kernel switches on weight >= 0.5 only, because on hardware
SFPMAD's product overflows before the add and |weight - 1| > |weight| for weight <= -0.5.

Run: python3 tests/ttnn/unit_tests/operations/eltwise/lerp_sfpu_numpy_port.py
Needs only numpy.
"""

from fractions import Fraction

import numpy as np

F32 = np.float32


def mad(x, y, z, fused=True):
    """SFPMAD: x * y + z in fp32. The fp32 product is exact in fp64, so one fp64 add then one
    rounding to fp32 models the fused instruction (up to rare double rounding). The hardware
    product overflows at 2^128 before the add; no domain below comes near that."""
    if fused:
        return (x.astype(np.float64) * y.astype(np.float64) + z.astype(np.float64)).astype(F32)
    return (x * y).astype(F32) + z


def to_bf16(x):
    """float32_to_bf16_rne, NaN handling omitted (no NaN inputs here)."""
    u = x.astype(F32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(F32)


def lerp_old(a, b, w, fused=True):
    # sfpi::vFloat result = in0 + in2 * (in1 - in0);
    d = b - a
    return mad(w, d, a, fused)


def lerp_new(a, b, w, fused=True):
    # sfpi::vFloat diff = in1 - in0;
    # v_if(in2 >= 0.5f) { in2 = in2 - 1.0f; in0 = in1; } v_endif;
    # sfpi::vFloat result = in0 + in2 * diff;
    d = b - a
    upper = w >= F32(0.5)
    coeff = np.where(upper, w - F32(1.0), w).astype(F32)
    base = np.where(upper, b, a).astype(F32)
    return mad(coeff, d, base, fused)


def lerp_torch_scalar(a, b, w, fused=True):
    """aten's scalar lerp: |w| < 0.5 ? a + w * (b - a) : b - (b - a) * (1 - w), unfused."""
    d = b - a
    lo = (a + (w * d).astype(F32)).astype(F32)
    hi = (b - (d * (F32(1.0) - w)).astype(F32)).astype(F32)
    return np.where(np.abs(w) < F32(0.5), lo, hi).astype(F32)


def exact(a, b, w, out_bf16=False):
    out = np.empty(a.shape, dtype=F32)
    for i, (ai, bi, wi) in enumerate(zip(a.tolist(), b.tolist(), w.tolist())):
        fa = Fraction(ai)
        out[i] = F32(float(fa + Fraction(wi) * (Fraction(bi) - fa)))
    return to_bf16(out) if out_bf16 else out


def ulp_err(actual, ref, out_bf16=False):
    ref = ref.astype(np.float64)
    spacing = np.spacing(np.abs(ref).astype(F32)).astype(np.float64)
    if out_bf16:
        spacing *= 2.0**16
    spacing = np.maximum(spacing, 2.0**-126)  # device flushes subnormals
    return np.abs(actual.astype(np.float64) - ref) / spacing


def rel_err(actual, ref):
    ref = ref.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(actual.astype(np.float64) - ref) / np.abs(ref)
    return np.nan_to_num(r, nan=0.0, posinf=0.0)


def report(name, a, b, w, out_bf16=False, fused=True):
    ref = exact(a, b, w, out_bf16)
    rows = []
    for label, fn in (("old", lerp_old), ("new", lerp_new), ("torch", lerp_torch_scalar)):
        got = fn(a, b, w, fused)
        if out_bf16:
            got = to_bf16(got)
        u = ulp_err(got, ref, out_bf16)
        r = rel_err(got, ref)
        k = int(np.argmax(u))
        rows.append(
            f"  {label:>5}: max ULP {u.max():12.1f}  max rel {r.max():9.2e}  >1 ULP {np.mean(u > 1) * 100:6.2f}%"
            f"  worst a={a[k]!r} b={b[k]!r} w={w[k]!r} -> {got[k]!r} (exact {ref[k]!r})"
        )
    print(f"{name} [{'bf16 out' if out_bf16 else 'fp32'}, n={a.size}, {'fused' if fused else 'unfused'} MAD]")
    print("\n".join(rows))


def main():
    rng = np.random.default_rng(0)
    n = 20000

    def mags(lo, hi, size):
        return (10.0 ** rng.uniform(lo, hi, size) * rng.choice([-1.0, 1.0], size)).astype(F32)

    print("Spot values (fp32):")
    for a, b, w in ((1000.0, 0.1, 1.0), (1.0e8, 1.0, 1.0), (300.0, 0.3, 0.999), (-65536.0, 1.0e-3, 0.75)):
        a, b, w = (np.array([v], dtype=F32) for v in (a, b, w))
        print(
            f"  lerp({a[0]!r}, {b[0]!r}, {w[0]!r}): old {lerp_old(a, b, w)[0]!r}  new {lerp_new(a, b, w)[0]!r}"
            f"  exact {exact(a, b, w)[0]!r}"
        )
    print()

    # 1. weight == 1 must return end; |start| >> |end| is where the old form loses it.
    a = mags(-2, 6, n)
    b = mags(-3, 1, n)
    w = np.ones(n, dtype=F32)
    report("weight == 1, |start| in [1e-2, 1e6], |end| in [1e-3, 10]", a, b, w)
    report("weight == 1, |start| in [1e-2, 1e6], |end| in [1e-3, 10]", to_bf16(a), to_bf16(b), w, out_bf16=True)

    # 2. weight in [0.5, 1], same magnitudes.
    w = rng.uniform(0.5, 1.0, n).astype(F32)
    report("weight in [0.5, 1], |start| in [1e-2, 1e6], |end| in [1e-3, 10]", a, b, w)
    report("same, unfused MAD", a, b, w, fused=False)

    # 3. Ordinary domain: start, end in [-100, 100], weight in [0, 1].
    a = rng.uniform(-100, 100, n).astype(F32)
    b = rng.uniform(-100, 100, n).astype(F32)
    w = rng.uniform(0.0, 1.0, n).astype(F32)
    report("start, end in [-100, 100], weight in [0, 1]", a, b, w)

    # 4. weight < 0.5 (negative weights included) takes the old expression unchanged: bit-identical.
    w = rng.uniform(-8.0, 0.4999, n).astype(F32)
    same = np.array_equal(lerp_old(a, b, w).view(np.uint32), lerp_new(a, b, w).view(np.uint32))
    print(f"weight in [-8, 0.5): old and new bit-identical: {same}")

    # 5. Endpoint property over magnitudes: fraction of lanes where lerp(a, b, 1) != b.
    a = mags(-6, 30, n)
    b = mags(-6, 6, n)
    w = np.ones(n, dtype=F32)
    for label, fn in (("old", lerp_old), ("new", lerp_new)):
        print(f"lerp(a, b, 1) != b, |a| in [1e-6, 1e30]: {label} {np.mean(fn(a, b, w) != b) * 100:.2f}%")


if __name__ == "__main__":
    main()
