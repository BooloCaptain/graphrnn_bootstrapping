#!/usr/bin/env python
"""GraphRNN pipeline entrypoint."""

import argparse
import os

# Parse arguments BEFORE importing torch to set CUDA device
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--cuda", type=int, default=0, help="CUDA device")
parser.add_argument("--cpu", action="store_true", help="Force CPU")
early_args, _ = parser.parse_known_args()

if not early_args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(early_args.cuda)

from graphrnn_clean.pipeline import build_pipeline_parser, run_pipeline


def main():
    parser = build_pipeline_parser()
    args = parser.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()