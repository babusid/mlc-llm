import math
import torch
import numpy as np
import tvm
from tvm import dlight as dl
from tvm import relax, te, tir
from typing import Any, Callable, Dict, Optional, Tuple

# max_seq_len = 11
# head_dim = 128
# theta = 500000.0
# factor = 16
# low_freq_factor = 1
# high_freq_factor = 1
# original_max_position_embeddings = 8192

max_seq_len = 30
head_dim = 128
partial_rotary_factor = 0.75
rotary_dim = int(head_dim * partial_rotary_factor)
theta = 10000.0
original_max_position_embeddings = 4096
max_position_embeddings = 131072
dtype = "float32"
rope_ext_factors = [
    1,
    1.118320672,
    1.250641126,
    1.398617824,
    1.564103225,
    1.74916897,
    1.956131817,
    2.187582649,
    2.446418898,
    2.735880826,
    3.059592084,
    3.421605075,
    3.826451687,
    4.279200023,
    4.785517845,
    5.351743533,
    5.984965424,
    6.693110555,
    7.485043894,
    8.370679318,
    9.36110372,
    10.4687158,
    11.70738129,
    13.09260651,
    14.64173252,
    16.37415215,
    18.31155283,
    20.47818807,
    22.90118105,
    25.61086418,
    28.64115884,
    32.03,
    32.1,
    32.13,
    32.23,
    32.6,
    32.61,
    32.64,
    32.66,
    32.7,
    32.71,
    32.93,
    32.97,
    33.28,
    33.49,
    33.5,
    44.16,
    47.77,
]


def compute_inv_freq(ext_factors_buf):
    """
    Compute inverse frequency for RoPE with ext_factors as a buffer parameter.
    ext_factors_buf: a te.placeholder representing the ext_factors array
    """

    def rope_freq_longrope(s: tir.Var, d: tir.Var):
        """Compute the inverse frequency of RoPE for longrope scaling."""
        d_range = rotary_dim

        scale = max_position_embeddings / original_max_position_embeddings
        scaling_factor = (
            math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))
            if scale > 1.0
            else 1.0
        )
        divisor = tir.power(theta, d * 2 % d_range / tir.const(d_range, "float32"))

        divisor = ext_factors_buf[d % (d_range // 2)] * divisor

        freq = s / divisor
        cos_freq = (tir.cos(freq) * scaling_factor).astype(dtype)
        sin_freq = (tir.sin(freq) * scaling_factor).astype(dtype)
        return cos_freq, sin_freq, freq.astype(dtype)

    return te.compute((max_seq_len, rotary_dim), rope_freq_longrope, name="inv_freq")


def main():
    bb = relax.BlockBuilder()

    # Create ext_factors as a relax constant
    ext_factors_np = np.array(rope_ext_factors, dtype="float32")
    ext_factors_const = relax.const(ext_factors_np, dtype="float32")

    with bb.function("main", params=[]):
        with bb.dataflow():
            # Pass ext_factors_const as an argument to the TE function
            # The compute_inv_freq function takes ext_factors_buf as a parameter
            inv_freq = bb.emit_te(compute_inv_freq, ext_factors_const)
            output = bb.emit_output(inv_freq)
        bb.emit_func_output(output)
    mod = bb.finalize()
    mod.show()

    dev = tvm.metal()
    target = tvm.target.Target.from_device(dev)

    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)

    exec = tvm.compile(mod, target=target)
    vm = relax.VirtualMachine(exec, dev)

    cos, sin, freq = vm["main"]()
    cos_np = cos.numpy()
    sin_np = sin.numpy()
    freq_np = freq.numpy()
    print(f"{cos_np=}")
    print(f"{sin_np=}")
    print(f"{freq_np=}")
    np.savez(
        "/Users/sidhartb/Work/mlc-llm/dist/debug/debug-tvm-rope-scaling-factors/tvm_rope_freq_longrope.npz",
        cos=cos_np,
        sin=sin_np,
        freq=freq_np,
    )
    return cos_np, sin_np


if __name__ == "__main__":
    mlc_inv_freq = main()
