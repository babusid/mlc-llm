import numpy as np
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest two NPZ files, check the last tensor in each for differences, and progressively tighten assertions."
    )
    parser.add_argument(
        "--file1", 
        type=str,
        required=False,
        help="Path to first input NPZ file.",
        default="/Users/sidhartb/Work/mlc-llm/dist/debug/debug-Phi-4-mini-instruct-hf/f1_tensor_dump_phi3_rotary_embedding.npz")
    parser.add_argument(
        "--file2", 
        type=str,
        required=False,
        help="Path to second input NPZ file.",
        default="/Users/sidhartb/Work/mlc-llm/dist/debug/debug-tvm-rope-scaling-factors/tvm_rope_freq_longrope.npz"
    )

    args = parser.parse_args()

    _f1 = np.load(args.file1)
    f1 = _f1[_f1.files[2]]

    _f2 = np.load(args.file2)
    f2 = _f2[_f2.files[0]]
    # # add batch dim of 1 to f2
    # if f1.ndim == f2.ndim + 1:
    #     f2 = np.expand_dims(f2, axis=0)
    

    assert f1.shape == f2.shape, f"Shape mismatch, tensors not comparable: {f1.shape} vs {f2.shape}"

    print(f"Num datapoints: {f1.shape}")

    # assertion thresholds to progressively tighten
    thresholds = [1e-3, 1e-4, 1e-5, 1e-6]

    for threshold in thresholds:
        try:
            np.testing.assert_allclose(f1, f2, rtol=threshold, atol=threshold)
        except AssertionError as e:
            print(e)
            exit()
        print(f"Assertion passed for threshold: {threshold}")
