import numpy as np
import networkx as nx


def bfs_seq(graph: nx.Graph, start_id: int) -> list[int]:
    dictionary = dict(nx.bfs_successors(graph, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next_nodes = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                next_nodes += neighbor
        output += next_nodes
        start = next_nodes
    return output


def encode_adj(adj: np.ndarray, max_prev_node: int, is_full: bool = False) -> np.ndarray:
    if is_full:
        max_prev_node = adj.shape[0] - 1

    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n - 1]

    adj_output = np.zeros((adj.shape[0], max_prev_node), dtype=np.float32)
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i, :] = adj_output[i, :][::-1]

    return adj_output


def decode_adj(adj_output: np.ndarray) -> np.ndarray:
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]), dtype=np.float32)
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i, ::-1][output_start:output_end]

    adj_full = np.zeros((adj_output.shape[0] + 1, adj_output.shape[0] + 1), dtype=np.float32)
    n = adj_full.shape[0]
    adj_full[1:n, 0:n - 1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T
    return adj_full


def graph_from_adj(adj: np.ndarray) -> nx.Graph:
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    return nx.from_numpy_array(adj)
