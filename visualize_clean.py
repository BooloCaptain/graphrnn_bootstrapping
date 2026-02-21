import argparse
from pathlib import Path
import pickle

from graphrnn_clean.visualize import draw_graph_list


def load_graph_list(path: Path):
    with open(path, "rb") as file:
        return pickle.load(file)


def main():
    parser = argparse.ArgumentParser(description="Visualize GraphRNN graph lists")
    parser.add_argument("--input", required=True, help="Path to .dat pickle file with graph list")
    parser.add_argument("--rows", type=int, default=4, help="Grid rows")
    parser.add_argument("--cols", type=int, default=4, help="Grid cols")
    parser.add_argument("--output", default="figures/clean_preview", help="Output image path prefix")
    parser.add_argument("--layout", default="spring", choices=["spring", "spectral"], help="Layout type")
    args = parser.parse_args()

    graphs = load_graph_list(Path(args.input))
    count = min(len(graphs), args.rows * args.cols)
    draw_graph_list(graphs[:count], args.rows, args.cols, args.output, layout=args.layout)


if __name__ == "__main__":
    main()
