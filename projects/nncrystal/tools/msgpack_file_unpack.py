import argparse
import os

from utils import load_images_entries, msgpack_data_index
import zipfile

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--file", required=True)
    parser.add_argument("--output")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)

    args = parser.parse_args()

    index = msgpack_data_index(args.file)

    start = args.start if args.start else 0
    end = args.end if args.end else len(index)

    entries = load_images_entries(args.file, list(range(start, end)), index_list=index)

    output = args.output if args.output else os.path.join(os.path.dirname(args.file), "unpacked.zip")
    with zipfile.ZipFile(output, "w") as zf:
        for entry in entries:
            zf.writestr(entry.get(b"name").decode("utf-8"), entry.get(b"data"))

