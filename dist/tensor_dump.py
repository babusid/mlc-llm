import numpy as np
import argparse


def dump_tensor_file(in_file, out_file):
    """Takes in an NPZ, and dumps the raw tensors in order to a text file."""
    data = np.load(in_file)
    with open(out_file, "w", encoding="utf-8") as wf:
        for filename in data.files:
            array = data[filename]
            wf.write(f"=== {filename} ===\n")
            wf.write(f"Shape: {array.shape}\n")
            wf.write(f"Dtype: {array.dtype}\n")
            wf.write("Data:\n")
            # Set print options to show full array without truncation
            with np.printoptions(threshold=np.inf, linewidth=np.inf):
                wf.write(f"{array}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump NPZ tensor shapes and data to text files.")
    parser.add_argument(
        "in_file", type=str, help="Input NPZ file or directory containing NPZ files."
    )
    parser.add_argument("out_file", type=str, help="Output text file or directory for dumped data.")
    args = parser.parse_args()

    import os

    _in_file = args.in_file
    _out_file = args.out_file

    if os.path.isfile(_in_file):
        if os.path.isdir(_out_file):
            raise ValueError("If input is a file, output must also be a file.")
        dump_tensor_file(_in_file, _out_file)
    elif os.path.isdir(_in_file):
        if not os.path.exists(_out_file):
            os.makedirs(_out_file)
        elif os.path.isfile(_out_file):
            raise ValueError("If input is a directory, output must also be a directory.")

        npz_files = [f for f in os.listdir(_in_file) if f.endswith(".npz")]

        def extract_number(filename):
            ''' Extract number from filename like "f123.npz" -> 123 '''
            try:
                return int(filename.split(".")[0][1:].split("_")[0])
            except (ValueError, IndexError):
                return float("inf")  # Put invalid filenames at the end

        npz_files.sort(key=extract_number)
        for f in npz_files:
            dump_tensor_file(
                os.path.join(_in_file, f), os.path.join(_out_file, f.replace(".npz", ".txt"))
            )
