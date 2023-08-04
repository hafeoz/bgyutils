#!/usr/bin/env python3

from argparse import ArgumentParser

from creation_wizard import creation_wizard
import structs
from slice import show_slice, SliceOrientation


def main():
    # https://gist.github.com/djwbrown/3e24bf4e0c5e9ee156a5?permalink_comment_id=4128738#gistcomment-4128738
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = ArgumentParser("Utilities for creating and inspecting BGY files")
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")
    parser.add_argument("bgyfile", help="BGY file")
    subparsers.required = True

    subparsers.add_parser("info", help="Show information about the BGY file")

    subparsers.add_parser("new", help="Create a new BGY file using the creation wizard")

    slice_subparser = subparsers.add_parser("slice", help="Slice a BGY file")
    slice_subparser.add_argument(
        "orientation", choices=["x", "i", "d"], help="Orientation"
    )
    slice_subparser.add_argument("idx", type=int, help="Index")

    args = parser.parse_args()
    if args.command == "info":
        with open(args.bgyfile, "rb") as f:
            bgy_file = structs.Bgy.from_io(f)
        print(bgy_file.description())
    elif args.command == "new":
        creation_wizard(args.bgyfile)
    elif args.command == "slice":
        if args.orientation == "x":
            orientation = SliceOrientation.XLine
        elif args.orientation == "i":
            orientation = SliceOrientation.Inline
        elif args.orientation == "d":
            orientation = SliceOrientation.Depth
        else:
            raise ValueError("Invalid orientation")
        with open(args.bgyfile, "rb") as f:
            bgy_file = structs.Bgy.from_io(f)
        for idx, data_packet in enumerate(bgy_file.data_packets):
            print(f"Trying to slice data packet {idx}...")
            show_slice(data_packet, orientation, args.idx)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
