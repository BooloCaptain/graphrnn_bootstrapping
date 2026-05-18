import argparse
import math
import pickle
import random
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch

from .config import ExperimentConfig
from .dataset import GraphSequenceDataset
from .eval_stats import clustering_stats, degree_stats, orbit_stats_all
from .generator import sample_graphs_rnn
from .graph_data import create_default_graphs, get_supported_datasets, split_graphs
from .model_core import GRUPlain
from .report_generator import generate_html_report, save_evaluation_metrics
from .trainer import train
from .visualize import draw_graph_list

ModelFactory = Callable[[ExperimentConfig, torch.device], tuple[torch.nn.Module, torch.nn.Module]]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def build_pipeline_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run toggleable GraphRNN pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=3000, help="Training epochs")
    parser.add_argument("--eval-epoch", type=int, default=None, help="Epoch to evaluate")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--skip-report", action="store_true", help="Skip HTML report")
    parser.add_argument("--no-orbits", action="store_true", help="Disable ORCA orbit stats")
    parser.add_argument("--preload-to-gpu", action="store_true", help="Enable fast VRAM static preloading")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--batch-ratio", type=int, default=None, help="Override batch ratio")
    parser.add_argument("--num-workers", type=int, default=4, help="CPU workers (only active if NOT preloading)")
    parser.add_argument("--amp", type=str, default="off", choices=["off", "bf16", "fp16"], help="Mixed precision mode")
    parser.add_argument("--test-total-size", type=int, default=256, help="Graphs to generate")
    parser.add_argument("--sample-time", type=int, default=1, help="Sampling index")
    parser.add_argument("--graph-type", type=str, default="grid", choices=get_supported_datasets())
    parser.add_argument("--permutations-per-graph", type=int, default=1, help="Data augmentation by random BFS permutations")
    return parser


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


class CachedSamplePoolLoader:
    """Loads batches directly from pre-loaded GPU VRAM tensors with safe shuffling."""
    def __init__(self, x_pool, y_pool, len_pool, batch_size, num_samples_per_epoch):
        self.x_pool = x_pool
        self.y_pool = y_pool
        self.len_pool = len_pool
        self.batch_size = batch_size
        self.num_samples_per_epoch = num_samples_per_epoch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        pool_size = self.x_pool.size(0)
        batches_to_yield = math.ceil(self.num_samples_per_epoch / self.batch_size)

        for _ in range(batches_to_yield):
            # THE FIX: Draw a fresh, unique subset of size 'batch_size' for EVERY batch
            # This mathematically guarantees zero identical clones inside a single forward pass.
            batch_indices = torch.randperm(pool_size, device=self.x_pool.device)[:self.batch_size]
            
            yield {
                "x": self.x_pool.index_select(0, batch_indices),
                "y": self.y_pool.index_select(0, batch_indices),
                "len": self.len_pool.index_select(0, batch_indices),
            }

    def __len__(self) -> int:
        return math.ceil(self.num_samples_per_epoch / self.batch_size)


def load_augmented_dataset_to_vram(dataset: GraphSequenceDataset, device: torch.device, permutations: int):
    """Bypasses CPU DataLoader by generating augmented views and shoving them into VRAM."""
    all_samples = []
    print(f"    Pre-calculating {permutations} random BFS permutations per graph...")
    
    # Looping over the dataset multiple times naturally generates new random BFS paths
    for view_idx in range(permutations):
        for i in range(len(dataset)):
            all_samples.append(dataset[i])
            
    x_pool = torch.stack([sample["x"] for sample in all_samples], dim=0).float().to(device, non_blocking=True)
    y_pool = torch.stack([sample["y"] for sample in all_samples], dim=0).float().to(device, non_blocking=True)
    len_pool = torch.stack([sample["len"] for sample in all_samples], dim=0).to(device, non_blocking=True)
    
    return x_pool, y_pool, len_pool


def clean_graphs(graph_real, graph_pred):
    random.shuffle(graph_real)
    random.shuffle(graph_pred)
    real_sizes = [len(graph) for graph in graph_real]
    pred_sizes = [len(graph) for graph in graph_pred]

    pred_graph_new = []
    for value in real_sizes:
        idx = min(range(len(pred_sizes)), key=lambda i: abs(pred_sizes[i] - value))
        pred_graph_new.append(graph_pred[idx])
    return graph_real, pred_graph_new


def default_model_factory(config: ExperimentConfig, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
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

    return rnn, output


def _build_train_loader(config: ExperimentConfig, args, dataset: GraphSequenceDataset, device: torch.device):
    if args.preload_to_gpu and device.type == "cuda":
        print(f"  [Mode: Fast VRAM] Preloading static graphs to GPU (Permutations: {args.permutations_per_graph})...")

        # The total pool size is now original graphs * permutations
        total_augmented_samples = len(dataset) * args.permutations_per_graph

        if config.batch_size > total_augmented_samples:
            print(f"\n  [!] WARNING: Batch size ({config.batch_size}) exceeds total augmented samples ({total_augmented_samples}).")
            print("      In VRAM preload mode, this will cause identical clones and NaN explosions!")

        x_pool, y_pool, len_pool = load_augmented_dataset_to_vram(dataset, device, args.permutations_per_graph)
        num_samples_per_epoch = config.batch_size * config.batch_ratio
        
        return CachedSamplePoolLoader(
            x_pool=x_pool,
            y_pool=y_pool,
            len_pool=len_pool,
            batch_size=config.batch_size,
            num_samples_per_epoch=num_samples_per_epoch,
        )

    print("  [Mode: Strict Paper] Using dynamic CPU DataLoader for infinite random BFS...")
    sample_strategy = torch.utils.data.WeightedRandomSampler(
        weights=[1.0 / len(dataset)] * len(dataset),
        num_samples=config.batch_size * config.batch_ratio,
        replacement=True,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        sampler=sample_strategy,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )


def _run_evaluation(config: ExperimentConfig, args, graphs_test, graphs_validate, reports_dir: Path, eval_epoch: int):
    eval_results = {}
    if args.skip_eval:
        print("\n[4/6] Evaluate (SKIPPED)")
        return eval_results

    print(f"\n[4/6] Evaluate (epoch {eval_epoch})")
    pred_path = config.graph_save_path / f"{config.fname_pred}{eval_epoch}_{args.sample_time}.dat"
    if not pred_path.exists():
        print("  Skipping evaluation: Generated graphs not found.")
        return eval_results

    graph_pred = load_graph_list(pred_path)
    graph_test_list, graph_pred_clean = clean_graphs(list(graphs_test), graph_pred)
    graph_val_list, _ = clean_graphs(list(graphs_validate), graph_pred)

    mmd_degree = degree_stats(graph_test_list, graph_pred_clean, is_parallel=False)
    mmd_clustering = clustering_stats(graph_test_list, graph_pred_clean, is_parallel=False)
    print(f"  Test Metrics: Degree MMD: {mmd_degree:.6f} | Clustering MMD: {mmd_clustering:.6f}")
    eval_results["test"] = {"degree_mmd": float(mmd_degree), "clustering_mmd": float(mmd_clustering)}

    if not args.no_orbits:
        try:
            mmd_orbits = orbit_stats_all(graph_test_list, graph_pred_clean, is_parallel=False)
            print(f"  Orbit MMD: {mmd_orbits:.6f}")
            eval_results["test"]["orbit_mmd"] = float(mmd_orbits)
        except Exception:
            pass

    mmd_degree_val = degree_stats(graph_val_list, graph_pred_clean, is_parallel=False)
    mmd_clustering_val = clustering_stats(graph_val_list, graph_pred_clean, is_parallel=False)
    eval_results["validation"] = {"degree_mmd": float(mmd_degree_val), "clustering_mmd": float(mmd_clustering_val)}

    save_evaluation_metrics(
        reports_dir / f"metrics_epoch{eval_epoch}.json",
        test_metrics=eval_results.get("test"),
        validation_metrics=eval_results.get("validation"),
    )
    return eval_results


def _run_visualization(config: ExperimentConfig, args, figures_dir: Path, eval_epoch: int):
    generated_viz_path = None
    if args.skip_viz:
        print("\n[5/6] Visualize (SKIPPED)")
        return generated_viz_path

    print(f"\n[5/6] Visualize (epoch {eval_epoch})")
    pred_path = config.graph_save_path / f"{config.fname_pred}{eval_epoch}_{args.sample_time}.dat"
    if pred_path.exists():
        graph_pred = load_graph_list(pred_path)
        output_path = figures_dir / f"pipeline_epoch{eval_epoch}_sample{args.sample_time}"
        generated_viz_path = Path(str(output_path) + ".png")
        draw_graph_list(graph_pred[:16], 4, 4, str(output_path), layout="spring")
        print(f"  ✓ Saved to {generated_viz_path}")
    return generated_viz_path


def _run_report(config: ExperimentConfig, args, device: torch.device, reports_dir: Path, eval_epoch: int, eval_results, training_viz_path: Path, generated_viz_path: Path | None):
    if args.skip_report:
        print("\n[6/6] Generate HTML Report (SKIPPED)")
        return

    print("\n[6/6] Generate HTML Report")
    report_path = reports_dir / f"report_epoch{eval_epoch}.html"
    config_dict = {
        "graph_type": config.graph_type,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "device": str(device),
    }
    generate_html_report(
        output_path=report_path,
        experiment_config=config_dict,
        training_metrics=None,
        evaluation_metrics=eval_results if eval_results else None,
        training_viz_path=training_viz_path,
        generated_viz_path=generated_viz_path,
        title=f"GraphRNN Report - Epoch {eval_epoch}",
    )
    print(f"  ✓ Report: {report_path.absolute()}")


def run_pipeline(args, model_factory: ModelFactory | None = None, loss_fn: LossFn | None = None):
    print("=" * 80)
    print("GraphRNN Clean Pipeline")
    print("=" * 80)

    reports_dir = Path("reports")
    figures_dir = Path("figures")
    reports_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    print("\n[1/6] Configuration")
    config = ExperimentConfig()
    config.graph_type = args.graph_type
    config.epochs = args.epochs
    config.test_total_size = args.test_total_size
    config.amp_mode = args.amp

    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.batch_ratio is not None:
        config.batch_ratio = args.batch_ratio

    seed_all(config.seed)

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda:0")
    print(f"  Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}, Batch ratio: {config.batch_ratio}")
    print(f"  Mode: {'VRAM Preload (Fast)' if args.preload_to_gpu else 'Dynamic CPU Loader (Strict Paper)'}")

    print("\n[2/6] Load Data")
    graphs_all, _ = create_default_graphs(config.graph_type)
    config.max_prev_node = 40
    graphs_train, graphs_validate, graphs_test = split_graphs(graphs_all, seed=config.seed)
    config.max_num_node = max(graph.number_of_nodes() for graph in graphs_all)

    config.graph_save_path.mkdir(parents=True, exist_ok=True)
    save_graph_list(graphs_train, config.graph_save_path / "train_split.dat")
    save_graph_list(graphs_validate, config.graph_save_path / "validate_split.dat")
    save_graph_list(graphs_test, config.graph_save_path / "test_split.dat")

    training_viz_base = figures_dir / "pipeline_training_data"
    draw_graph_list(graphs_train[:16], 4, 4, str(training_viz_base), layout="spring")
    training_viz_path = Path(str(training_viz_base) + ".png")

    model_builder = model_factory or default_model_factory

    if not args.skip_train:
        print("\n[3/6] Train Model")
        dataset = GraphSequenceDataset(
            graphs=graphs_train,
            max_num_node=config.max_num_node,
            max_prev_node=config.max_prev_node,
        )
        train_loader = _build_train_loader(config, args, dataset, device)
        rnn, output = model_builder(config, device)
        train(config=config, dataset_loader=train_loader, rnn=rnn, output=output, device=device, loss_fn=loss_fn)
        print("  ✓ Training complete")

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

    eval_epoch = args.eval_epoch if args.eval_epoch is not None else config.epochs
    eval_results = _run_evaluation(config, args, graphs_test, graphs_validate, reports_dir, eval_epoch)
    generated_viz_path = _run_visualization(config, args, figures_dir, eval_epoch)
    _run_report(config, args, device, reports_dir, eval_epoch, eval_results, training_viz_path, generated_viz_path)

    print("\n" + "=" * 80)
    print("Pipeline Complete")
    print("=" * 80)
