import numpy as np
import torch
import networkx as nx

from .sequence_codec import bfs_seq, encode_adj


class GraphSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, graphs: list[nx.Graph], max_num_node: int, max_prev_node: int):
        self.adj_all = [np.asarray(nx.to_numpy_array(graph), dtype=np.float32) for graph in graphs]
        self.len_all = [graph.number_of_nodes() for graph in graphs]
        self.max_num_node = max_num_node
        self.max_prev_node = max_prev_node

    def __len__(self) -> int:
        return len(self.adj_all)

    def __getitem__(self, idx: int):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros((self.max_num_node, self.max_prev_node), dtype=np.float32)
        y_batch = np.zeros((self.max_num_node, self.max_prev_node), dtype=np.float32)

        x_batch[0, :] = 1.0
        length = adj_copy.shape[0]

        permute_idx = np.random.permutation(length)
        adj_copy = adj_copy[np.ix_(permute_idx, permute_idx)]

        graph_perm = nx.from_numpy_array(adj_copy)
        start_idx = np.random.randint(length)
        bfs_idx = np.array(bfs_seq(graph_perm, start_idx))
        adj_copy = adj_copy[np.ix_(bfs_idx, bfs_idx)]

        adj_encoded = encode_adj(adj_copy, max_prev_node=self.max_prev_node)
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded

        return {
            "x": torch.from_numpy(x_batch),
            "y": torch.from_numpy(y_batch),
            "len": torch.tensor(length, dtype=torch.long),
        }
