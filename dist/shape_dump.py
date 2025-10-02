import numpy as np
import os
import argparse


def dump_file_shapes(in_file: os.PathLike, out_file: os.PathLike):
    """
    Read a .npz file, and dump the shape of its arrays into a single
    line in the out_file.
    """
    np_buf = np.load(in_file)
    shapes = {filename: np_buf[filename].shape for filename in np_buf.files}
    with open(out_file, "a", encoding="utf-8") as file_handle:
        file_handle.write(f"{in_file}:\n")
        for k, v in shapes.items():
            file_handle.write(f"{k}: {v} ")
        file_handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump the shapes of arrays in a .npz file to a text file."
    )
    parser.add_argument("in_file", type=str, help="Path to the directory with all the .npz files.")
    parser.add_argument("out_file", type=str, help="Path to the output text file.")
    args = parser.parse_args()
    in_file = args.in_file
    out_file = args.out_file

    if os.path.exists(out_file):
        os.remove(out_file)

    # Get all .npz files and sort them by numeric suffix
    npz_files = [f for f in os.listdir(in_file) if f.endswith(".npz")]

    # Sort by extracting the numeric part after 'f'
    def extract_number(filename):
        # Extract number from filename like "f123.npz" -> 123
        try:
            return int(filename.split(".")[0][1:].split("_")[0])
        except (ValueError, IndexError):
            return float("inf")  # Put invalid filenames at the end

    npz_files.sort(key=extract_number)
    for f in npz_files:
        dump_file_shapes(os.path.join(in_file, f), out_file)
