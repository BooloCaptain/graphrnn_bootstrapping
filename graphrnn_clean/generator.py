import torch
import networkx as nx

from .model_core import sample_sigmoid
from .sequence_codec import decode_adj, graph_from_adj


@torch.no_grad()
def sample_graphs_rnn(
    rnn: torch.nn.Module,
    output_head: torch.nn.Module,
    num_graphs: int,
    max_num_node: int,
    max_prev_node: int,
    num_layers: int,
    device: torch.device,
) -> list[nx.Graph]:
    rnn.eval()
    output_head.eval()

    graphs: list[nx.Graph] = []
    batch_size = min(32, num_graphs)

    while len(graphs) < num_graphs:
        current_batch = min(batch_size, num_graphs - len(graphs))
        rnn.hidden = rnn.init_hidden(current_batch, device)

        y_pred_long = torch.zeros(current_batch, max_num_node, max_prev_node, device=device)
        x_step = torch.ones(current_batch, 1, max_prev_node, device=device)

        for i in range(max_num_node):
            h = rnn(x_step)
            hidden_null = torch.zeros(num_layers - 1, h.size(0), h.size(2), device=device)
            output_head.hidden = torch.cat((h.permute(1, 0, 2), hidden_null), dim=0)
            x_step = torch.zeros(current_batch, 1, max_prev_node, device=device)
            output_x_step = torch.ones(current_batch, 1, 1, device=device)

            for j in range(min(max_prev_node, i + 1)):
                output_y_pred_step = output_head(output_x_step)
                output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
                x_step[:, :, j : j + 1] = output_x_step
                output_head.hidden = output_head.hidden.detach()

            y_pred_long[:, i : i + 1, :] = x_step
            rnn.hidden = rnn.hidden.detach()

        y_pred_long_data = y_pred_long.long().cpu().numpy()
        for i in range(current_batch):
            adj_pred = decode_adj(y_pred_long_data[i])
            graphs.append(graph_from_adj(adj_pred))

    return graphs
