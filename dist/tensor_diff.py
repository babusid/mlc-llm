import numpy as np
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest two NPZ files, check the last tensor in each for differences, and progressively tighten assertions."
    )
    parser.add_argument("file1", type=str, help="Path to first input NPZ file.")
    parser.add_argument("file2", type=str, help="Path to second input NPZ file.")

    args = parser.parse_args()

    f1 = np.load(args.file1)
    f1 = f1[f1.files[-1]]

    f2 = np.load(args.file2)
    f2 = f2[f2.files[-1]]

    f1 = f1.flatten()
    f2 = f2.flatten()

    assert f1.shape == f2.shape, f"Shape mismatch, tensors not comparable: {f1.shape} vs {f2.shape}"

    print(f"Num datapoints: {f1.shape}")

    # assertion thresholds to progressively tighten
    thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    for threshold in thresholds:
        try:
            np.testing.assert_allclose(f1, f2, rtol=threshold, atol=threshold)
        except AssertionError as e:
            print(e)
            exit()
        print(f"Assertion passed for threshold: {threshold}")
