import argparse
from pathlib import Path
from random import shuffle

import pickle

from graphrnn_clean.config import ExperimentConfig
from graphrnn_clean.eval_stats import degree_stats, clustering_stats, orbit_stats_all
from graphrnn_clean.graph_data import create_default_graphs, split_graphs


def load_graph_list(path: Path):
    with open(path, "rb") as file:
        return pickle.load(file)


def clean_graphs(graph_real, graph_pred):
    shuffle(graph_real)
    shuffle(graph_pred)

    real_sizes = [len(graph_real[i]) for i in range(len(graph_real))]
    pred_sizes = [len(graph_pred[i]) for i in range(len(graph_pred))]

    pred_graph_new = []
    for value in real_sizes:
        idx = min(range(len(pred_sizes)), key=lambda i: abs(pred_sizes[i] - value))
        pred_graph_new.append(graph_pred[idx])
    return graph_real, pred_graph_new


def main():
    parser = argparse.ArgumentParser(description="Evaluate clean GraphRNN outputs")
    parser.add_argument("--graphs-dir", default="graphs", help="Directory with generated graph pickles")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch checkpoint to evaluate")
    parser.add_argument("--sample-time", type=int, default=1, help="Sampling index in filename")
    parser.add_argument("--no-clean", action="store_true", help="Skip size matching between real/pred")
    parser.add_argument("--orbits", action="store_true", help="Enable ORCA orbit stats (requires eval/orca/orca)")
    parser.add_argument("--no-emd", action="store_true", help="Use Gaussian kernel (no pyemd dependency)")
    parser.add_argument("--parallel", action="store_true", help="Enable multiprocessing for stats")
    args = parser.parse_args()

    config = ExperimentConfig()
    graphs_all, _ = create_default_graphs(config.graph_type)
    graphs_train, graphs_validate, graphs_test = split_graphs(graphs_all, seed=config.seed)

    graphs_dir = Path(args.graphs_dir)
    pred_path = graphs_dir / f"{config.fname_pred}{args.epoch}_{args.sample_time}.dat"
    if not pred_path.exists():
        raise FileNotFoundError(f"Generated graphs not found: {pred_path}")

    graph_pred = load_graph_list(pred_path)
    graph_test = list(graphs_test)
    graph_validate = list(graphs_validate)

    if args.no_clean:
        shuffle(graph_pred)
        graph_pred = graph_pred[: len(graph_test)]
    else:
        graph_test, graph_pred = clean_graphs(graph_test, graph_pred)

    mmd_degree = degree_stats(graph_test, graph_pred, is_parallel=args.parallel, use_emd=not args.no_emd)
    mmd_clustering = clustering_stats(graph_test, graph_pred, is_parallel=args.parallel, use_emd=not args.no_emd)
    print(f"degree_mmd: {mmd_degree}")
    print(f"clustering_mmd: {mmd_clustering}")

    if args.orbits:
        mmd_orbits = orbit_stats_all(graph_test, graph_pred, is_parallel=args.parallel)
        print(f"orbits_mmd: {mmd_orbits}")

    mmd_degree_val = degree_stats(graph_validate, graph_pred, is_parallel=args.parallel, use_emd=not args.no_emd)
    mmd_clustering_val = clustering_stats(graph_validate, graph_pred, is_parallel=args.parallel, use_emd=not args.no_emd)
    print(f"degree_mmd_validate: {mmd_degree_val}")
    print(f"clustering_mmd_validate: {mmd_clustering_val}")


if __name__ == "__main__":
    main()
