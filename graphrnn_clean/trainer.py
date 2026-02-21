import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import MultiStepLR

from .config import ExperimentConfig
from .generator import sample_graphs_rnn
from .model_core import binary_cross_entropy_weight


def _save_graph_list(graphs, fname: Path):
    with open(fname, "wb") as file:
        pickle.dump(graphs, file)


def train_rnn_epoch(
    epoch: int,
    config: ExperimentConfig,
    rnn: torch.nn.Module,
    output: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer_rnn: torch.optim.Optimizer,
    optimizer_output: torch.optim.Optimizer,
    scheduler_rnn: MultiStepLR,
    scheduler_output: MultiStepLR,
    device: torch.device,
) -> float:
    rnn.train()
    output.train()
    loss_sum = 0.0

    for batch_idx, data in enumerate(data_loader):
        optimizer_rnn.zero_grad()
        optimizer_output.zero_grad()

        x_unsorted = data["x"].float().to(device)
        y_unsorted = data["y"].float().to(device)
        y_len_unsorted = data["len"]
        y_len_max = int(y_len_unsorted.max().item())

        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0), device=device)

        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len_list = y_len.cpu().numpy().tolist()

        x = torch.index_select(x_unsorted, 0, sort_index.to(device))
        y = torch.index_select(y_unsorted, 0, sort_index.to(device))

        y_reshape = pack_padded_sequence(y, y_len_list, batch_first=True).data
        idx = torch.arange(y_reshape.size(0) - 1, -1, -1, device=device)
        y_reshape = y_reshape.index_select(0, idx)
        y_reshape = y_reshape.view(y_reshape.size(0), y_reshape.size(1), 1)

        output_x = torch.cat((torch.ones(y_reshape.size(0), 1, 1, device=device), y_reshape[:, 0:-1, 0:1]), dim=1)
        output_y = y_reshape

        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len_list))
        for i in range(len(output_y_len_bin) - 1, 0, -1):
            count_temp = np.sum(output_y_len_bin[i:])
            output_y_len.extend([min(i, y.size(2))] * int(count_temp))

        h = rnn(x, pack=True, input_len=y_len_list)
        h = pack_padded_sequence(h, y_len_list, batch_first=True).data
        idx = torch.arange(h.size(0) - 1, -1, -1, device=device)
        h = h.index_select(0, idx)

        hidden_null = torch.zeros(config.num_layers - 1, h.size(0), h.size(1), device=device)
        output.hidden = torch.cat((h.view(1, h.size(0), h.size(1)), hidden_null), dim=0)

        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = torch.sigmoid(y_pred)

        y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
        output_y = pack_padded_sequence(output_y, output_y_len, batch_first=True)
        output_y = pad_packed_sequence(output_y, batch_first=True)[0]

        loss = binary_cross_entropy_weight(y_pred, output_y)
        loss.backward()

        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        if epoch % config.epochs_log == 0 and batch_idx == 0:
            print(f"Epoch {epoch}/{config.epochs} - train loss: {loss.item():.6f}")

        feature_dim = y.size(1) * y.size(2)
        loss_sum += loss.item() * feature_dim

    return loss_sum / (batch_idx + 1)


def train(
    config: ExperimentConfig,
    dataset_loader: torch.utils.data.DataLoader,
    rnn: torch.nn.Module,
    output: torch.nn.Module,
    device: torch.device,
):
    config.model_save_path.mkdir(parents=True, exist_ok=True)
    config.graph_save_path.mkdir(parents=True, exist_ok=True)

    # Print GPU memory info
    if device.type == "cuda":
        print("\n" + "=" * 70)
        print("GPU Memory Monitoring")
        print("=" * 70)
        torch.cuda.reset_peak_memory_stats()
        print(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    optimizer_rnn = optim.Adam(rnn.parameters(), lr=config.lr)
    optimizer_output = optim.Adam(output.parameters(), lr=config.lr)
    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=config.milestones, gamma=config.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=config.milestones, gamma=config.lr_rate)

    for epoch in range(1, config.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        epoch_loss = train_rnn_epoch(
            epoch,
            config,
            rnn,
            output,
            dataset_loader,
            optimizer_rnn,
            optimizer_output,
            scheduler_rnn,
            scheduler_output,
            device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_start
        batch_size = dataset_loader.batch_size or config.batch_size
        samples_per_epoch = batch_size * len(dataset_loader)
        if epoch_seconds > 0:
            samples_s = samples_per_epoch / epoch_seconds
            samples_s_text = f"{samples_s:.2f} samples/s"
        else:
            samples_s_text = "n/a samples/s"

        # Print GPU memory periodically
        if device.type == "cuda" and epoch % config.epochs_log == 0:
            current_mem = torch.cuda.memory_allocated() / 1024**2
            max_mem = torch.cuda.max_memory_allocated() / 1024**2
            reserved_mem = torch.cuda.memory_reserved() / 1024**2
            print(
                f"Epoch {epoch} completed - avg loss: {epoch_loss:.6f} | {samples_s_text} | "
                f"GPU Memory: {current_mem:.2f} MB (Peak: {max_mem:.2f} MB, Reserved: {reserved_mem:.2f} MB)"
            )
        else:
            print(f"Epoch {epoch} completed - avg loss: {epoch_loss:.6f} | {samples_s_text}")

        if epoch % config.epochs_test == 0 and epoch >= config.epochs_test_start:
            generated = sample_graphs_rnn(
                rnn=rnn,
                output_head=output,
                num_graphs=config.test_total_size,
                max_num_node=config.max_num_node,
                max_prev_node=config.max_prev_node,
                num_layers=config.num_layers,
                device=device,
            )
            fname = config.graph_save_path / f"{config.fname_pred}{epoch}_1.dat"
            _save_graph_list(generated, fname)
            print(f"Saved generated graphs: {fname}")

        if config.save_checkpoints and epoch % config.epochs_save == 0:
            rnn_path = config.model_save_path / f"{config.fname}lstm_{epoch}.dat"
            out_path = config.model_save_path / f"{config.fname}output_{epoch}.dat"
            torch.save(rnn.state_dict(), rnn_path)
            torch.save(output.state_dict(), out_path)
            print(f"Saved checkpoints at epoch {epoch}")
