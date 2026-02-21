import argparse
import os
import random

import numpy as np
import torch

from graphrnn_clean.config import ExperimentConfig
from graphrnn_clean.dataset import GraphSequenceDataset
from graphrnn_clean.graph_data import create_default_graphs, split_graphs
from graphrnn_clean.model_core import GRUPlain
from graphrnn_clean.trainer import train


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Run clean GraphRNN default experiment")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--test-total-size", type=int, default=None, help="Generated graph count at test checkpoints")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--batch-ratio", type=int, default=None, help="Override batches sampled per epoch")
    args = parser.parse_args()

    config = ExperimentConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.test_total_size is not None:
        config.test_total_size = args.test_total_size
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.batch_ratio is not None:
        config.batch_ratio = args.batch_ratio
    config.cuda = args.cuda

    seed_all(config.seed)

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda)
        device = torch.device("cuda")

    graphs, default_max_prev = create_default_graphs(config.graph_type)
    config.max_prev_node = default_max_prev

    graphs_train, _, _ = split_graphs(graphs, seed=config.seed)
    config.max_num_node = max(graph.number_of_nodes() for graph in graphs)

    dataset = GraphSequenceDataset(
        graphs=graphs_train,
        max_num_node=config.max_num_node,
        max_prev_node=config.max_prev_node,
    )

    sample_strategy = torch.utils.data.WeightedRandomSampler(
        weights=[1.0 / len(dataset) for _ in range(len(dataset))],
        num_samples=config.batch_size * config.batch_ratio,
        replacement=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=sample_strategy,
    )

    rnn = GRUPlain(
        input_size=config.max_prev_node,
        embedding_size=config.embedding_size_rnn,
        hidden_size=config.hidden_size_rnn,
        num_layers=config.num_layers,
        has_input=True,
        has_output=True,
        output_size=config.hidden_size_rnn_output,
    ).to(device)

    output = GRUPlain(
        input_size=1,
        embedding_size=config.embedding_size_rnn_output,
        hidden_size=config.hidden_size_rnn_output,
        num_layers=config.num_layers,
        has_input=True,
        has_output=True,
        output_size=1,
    ).to(device)

    print(f"Running clean default experiment on device={device}")
    print(f"graphs(train)={len(graphs_train)}, max_num_node={config.max_num_node}, max_prev_node={config.max_prev_node}")

    train(config=config, dataset_loader=dataloader, rnn=rnn, output=output, device=device)


if __name__ == "__main__":
    main()
