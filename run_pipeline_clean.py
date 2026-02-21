#!/usr/bin/env python
"""
End-to-end GraphRNN pipeline: load → train → generate → evaluate → visualize

Orchestrates the complete workflow with automatic checkpointing and resumption.
"""

import argparse
import os
import sys
from pathlib import Path

# Parse arguments BEFORE importing torch to set CUDA device
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--cuda", type=int, default=0, help="CUDA device")
parser.add_argument("--cpu", action="store_true", help="Force CPU")
early_args, _ = parser.parse_known_args()

# Set CUDA device BEFORE importing torch
if not early_args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(early_args.cuda)

import pickle
import random
import subprocess

import numpy as np
import torch

from graphrnn_clean.config import ExperimentConfig
from graphrnn_clean.dataset import GraphSequenceDataset
from graphrnn_clean.generator import sample_graphs_rnn
from graphrnn_clean.graph_data import create_default_graphs, split_graphs, get_supported_datasets
from graphrnn_clean.model_core import GRUPlain
from graphrnn_clean.trainer import train
from graphrnn_clean.eval_stats import degree_stats, clustering_stats, orbit_stats_all
from graphrnn_clean.visualize import draw_graph_list
from graphrnn_clean.gpu_monitor import print_gpu_info, get_gpu_memory_stats
from graphrnn_clean.report_generator import (
    generate_html_report,
    save_evaluation_metrics
)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graph_list(path: Path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_graph_list(graphs, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(graphs, file)


def clean_graphs(graph_real, graph_pred):
    """Match generated graphs to real graph sizes."""
    random.shuffle(graph_real)
    random.shuffle(graph_pred)

    real_sizes = [len(g) for g in graph_real]
    pred_sizes = [len(g) for g in graph_pred]

    pred_graph_new = []
    for value in real_sizes:
        idx = min(range(len(pred_sizes)), key=lambda i: abs(pred_sizes[i] - value))
        pred_graph_new.append(graph_pred[idx])
    return graph_real, pred_graph_new


def main():
    supported_datasets = get_supported_datasets()
    parser = argparse.ArgumentParser(
        description="Run complete GraphRNN pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Supported graph types: {', '.join(supported_datasets)}"
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--eval-epoch", type=int, default=None, help="Epoch to evaluate (default: last trained)")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (use existing checkpoint)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--skip-report", action="store_true", help="Skip HTML report generation")
    parser.add_argument("--no-orbits", action="store_true", help="Disable ORCA orbit stats")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--batch-ratio", type=int, default=None, help="Override batch ratio")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of data loading workers")
    parser.add_argument("--test-total-size", type=int, default=256, help="Graphs to generate")
    parser.add_argument("--sample-time", type=int, default=1, help="Sampling index")
    parser.add_argument(
        "--graph-type",
        type=str,
        default="grid",
        choices=supported_datasets,
        help=f"Graph type for dataset (default: grid)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("GraphRNN Clean Pipeline")
    print("=" * 80)

    # Create output directories
    reports_dir = Path("reports")
    figures_dir = Path("figures")
    reports_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # =========================================================================
    # STEP 1: CONFIGURE
    # =========================================================================
    print("\n[1/6] Configuration")
    config = ExperimentConfig()
    config.graph_type = args.graph_type
    config.epochs = args.epochs
    config.test_total_size = args.test_total_size
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.batch_ratio is not None:
        config.batch_ratio = args.batch_ratio
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    seed_all(config.seed)
    
    # Determine device (CUDA_VISIBLE_DEVICES already set before torch import)
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        print(f"  Device: CPU")
    else:
        # CUDA_VISIBLE_DEVICES already set, just create cuda device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        print(f"  Device: CUDA (GPU {args.cuda}, {gpu_count} GPU(s) available)")
        if cuda_available:
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Verify CUDA is actually being used
        test_tensor = torch.randn(10, 10, device=device)
        if test_tensor.is_cuda:
            print(f"  ✓ CUDA tensor allocation verified")
        else:
            print(f"  ✗ WARNING: Tensor is on CPU despite CUDA device selection!")
    
    # Enable cuDNN autotuning for fixed input sizes
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Show detailed GPU info if available
    if not args.cpu and torch.cuda.is_available():
        print_gpu_info()

    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}, Batch ratio: {config.batch_ratio}, Workers: {config.num_workers}")
    print(f"  Test generation count: {config.test_total_size}")

    # =========================================================================
    # STEP 2: LOAD DATA
    # =========================================================================
    print("\n[2/6] Load Data")
    graphs_all, default_max_prev = create_default_graphs(config.graph_type)
    config.max_prev_node = default_max_prev
    graphs_train, graphs_validate, graphs_test = split_graphs(graphs_all, seed=config.seed)
    config.max_num_node = max(g.number_of_nodes() for g in graphs_all)

    print(f"  Graph type: {config.graph_type}")
    print(f"  Total graphs: {len(graphs_all)}")
    print(f"  Train: {len(graphs_train)}, Validate: {len(graphs_validate)}, Test: {len(graphs_test)}")
    print(f"  Max nodes: {config.max_num_node}, Max prev: {config.max_prev_node}")

    # Save splits for evaluation consistency
    config.graph_save_path.mkdir(parents=True, exist_ok=True)
    save_graph_list(graphs_train, config.graph_save_path / "train_split.dat")
    save_graph_list(graphs_validate, config.graph_save_path / "validate_split.dat")
    save_graph_list(graphs_test, config.graph_save_path / "test_split.dat")
    print(f"  Saved splits to {config.graph_save_path}/")
    
    # Visualize training data samples
    training_viz_base = figures_dir / "pipeline_training_data"
    num_train_samples = min(16, len(graphs_train))
    draw_graph_list(graphs_train[:num_train_samples], 4, 4, str(training_viz_base), layout="spring")
    # draw_graph_list adds .png extension, so actual file is .png.png
    training_viz_path = Path(str(training_viz_base) + ".png")
    print(f"  ✓ Training data visualized ({num_train_samples} samples)")

    # =========================================================================
    # STEP 3: TRAIN
    # =========================================================================
    if not args.skip_train:
        print("\n[3/6] Train Model")
        
        dataset = GraphSequenceDataset(
            graphs=graphs_train,
            max_num_node=config.max_num_node,
            max_prev_node=config.max_prev_node,
        )

        sample_strategy = torch.utils.data.WeightedRandomSampler(
            weights=[1.0 / len(dataset)] * len(dataset),
            num_samples=config.batch_size * config.batch_ratio,
            replacement=True,
        )

        dataloader_kwargs = {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "sampler": sample_strategy,
        }
        if device.type == "cuda":
            dataloader_kwargs["pin_memory"] = True
            dataloader_kwargs["persistent_workers"] = config.num_workers > 0
            dataloader_kwargs["prefetch_factor"] = 2 if config.num_workers > 0 else None

        dataloader = torch.utils.data.DataLoader(
            dataset,
            **{k: v for k, v in dataloader_kwargs.items() if v is not None},
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

        print(f"  Training for {config.epochs} epochs...")
        train(config=config, dataset_loader=dataloader, rnn=rnn, output=output, device=device)
        print(f"  ✓ Training complete")
        
        # Force generation at final epoch if not already done
        final_pred_path = config.graph_save_path / f"{config.fname_pred}{config.epochs}_{args.sample_time}.dat"
        if not final_pred_path.exists():
            print(f"  Generating graphs at final epoch {config.epochs}...")
            generated = sample_graphs_rnn(
                rnn=rnn,
                output_head=output,
                num_graphs=config.test_total_size,
                max_num_node=config.max_num_node,
                max_prev_node=config.max_prev_node,
                num_layers=config.num_layers,
                device=device,
            )
            save_graph_list(generated, final_pred_path)
            print(f"  ✓ Saved {len(generated)} generated graphs")
    else:
        print("\n[3/6] Train Model (SKIPPED)")

    # =========================================================================
    # STEP 4: EVALUATE
    # =========================================================================
    eval_epoch = args.eval_epoch if args.eval_epoch is not None else config.epochs
    eval_results = {}
    
    if not args.skip_eval:
        print(f"\n[4/6] Evaluate (epoch {eval_epoch})")
        
        pred_path = config.graph_save_path / f"{config.fname_pred}{eval_epoch}_{args.sample_time}.dat"
        if not pred_path.exists():
            print(f"  ✗ Generated graphs not found: {pred_path}")
            print(f"  Skipping evaluation")
        else:
            graph_pred = load_graph_list(pred_path)
            graph_test_list = list(graphs_test)
            graph_validate_list = list(graphs_validate)

            graph_test_list, graph_pred_clean = clean_graphs(graph_test_list, graph_pred)

            print(f"  Evaluating {len(graph_pred_clean)} generated graphs...")
            mmd_degree = degree_stats(graph_test_list, graph_pred_clean, is_parallel=False)
            mmd_clustering = clustering_stats(graph_test_list, graph_pred_clean, is_parallel=False)
            
            print(f"\n  Test Set Metrics:")
            print(f"    Degree MMD:      {mmd_degree:.6f}")
            print(f"    Clustering MMD:  {mmd_clustering:.6f}")
            
            # Store test metrics
            eval_results['test'] = {
                'degree_mmd': float(mmd_degree),
                'clustering_mmd': float(mmd_clustering)
            }

            if not args.no_orbits:
                try:
                    mmd_orbits = orbit_stats_all(graph_test_list, graph_pred_clean, is_parallel=False)
                    print(f"    Orbit MMD:       {mmd_orbits:.6f}")
                    eval_results['test']['orbit_mmd'] = float(mmd_orbits)
                except Exception as e:
                    print(f"    Orbit MMD:       FAILED ({e})")

            mmd_degree_val = degree_stats(graph_validate_list, graph_pred_clean, is_parallel=False)
            mmd_clustering_val = clustering_stats(graph_validate_list, graph_pred_clean, is_parallel=False)
            
            print(f"\n  Validation Set Metrics:")
            print(f"    Degree MMD:      {mmd_degree_val:.6f}")
            print(f"    Clustering MMD:  {mmd_clustering_val:.6f}")
            
            # Store validation metrics
            eval_results['validation'] = {
                'degree_mmd': float(mmd_degree_val),
                'clustering_mmd': float(mmd_clustering_val)
            }
            
            # Save evaluation metrics to JSON
            eval_metrics_path = reports_dir / f"metrics_epoch{eval_epoch}.json"
            save_evaluation_metrics(
                eval_metrics_path,
                test_metrics=eval_results.get('test'),
                validation_metrics=eval_results.get('validation')
            )
            
            print(f"\n  ✓ Evaluation complete")
    else:
        print(f"\n[4/6] Evaluate (SKIPPED)")

    # =========================================================================
    # STEP 5: VISUALIZE
    # =========================================================================
    generated_viz_path = None
    
    if not args.skip_viz:
        print(f"\n[5/6] Visualize (epoch {eval_epoch})")
        
        pred_path = config.graph_save_path / f"{config.fname_pred}{eval_epoch}_{args.sample_time}.dat"
        if not pred_path.exists():
            print(f"  ✗ Generated graphs not found: {pred_path}")
            print(f"  Skipping visualization")
        else:
            graph_pred = load_graph_list(pred_path)
            
            output_path = figures_dir / f"pipeline_epoch{eval_epoch}_sample{args.sample_time}"
            generated_viz_path = Path(str(output_path) + ".png")
            count = min(len(graph_pred), 16)
            draw_graph_list(graph_pred[:count], 4, 4, str(output_path), layout="spring")
            
            print(f"  Rendered {count} graphs")
            print(f"  ✓ Saved to {generated_viz_path}")
    else:
        print(f"\n[5/6] Visualize (SKIPPED)")

    # =========================================================================
    # STEP 6: GENERATE HTML REPORT
    # =========================================================================
    if not args.skip_report:
        print(f"\n[6/6] Generate HTML Report")
        
        report_path = reports_dir / f"report_epoch{eval_epoch}.html"
        
        # Prepare config dict for report
        config_dict = {
            'graph_type': config.graph_type,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'lr': config.lr,
            'hidden_size_rnn': config.hidden_size_rnn,
            'embedding_size_rnn': config.embedding_size_rnn,
            'max_prev_node': config.max_prev_node,
            'device': 'CPU' if args.cpu else f'CUDA {args.cuda}'
        }
        
        # Generate report
        generate_html_report(
            output_path=report_path,
            experiment_config=config_dict,
            training_metrics=None,  # Would need trainer.py modification to track
            evaluation_metrics=eval_results if eval_results else None,
            training_viz_path=training_viz_path,
            generated_viz_path=generated_viz_path,
            title=f"GraphRNN Experiment Report - Epoch {eval_epoch}"
        )
        
        print(f"  📄 Report: {report_path.absolute()}")
        print(f"  ✓ HTML report generated")
    else:
        print(f"\n[6/6] Generate HTML Report (SKIPPED)")

    print("\n" + "=" * 80)
    print("Pipeline Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

