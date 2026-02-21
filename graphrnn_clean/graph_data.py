import random
import networkx as nx


# Dataset configurations: (generator_function, max_prev_node)
DATASET_CONFIGS = {
    # Synthetic datasets - graphs with regular structures
    "grid": {
        "generator": lambda: _create_grid_graphs(),
        "max_prev_node": 40,
    },
    "grid_small": {
        "generator": lambda: _create_grid_small_graphs(),
        "max_prev_node": 15,
    },
    "grid_big": {
        "generator": lambda: _create_grid_big_graphs(),
        "max_prev_node": 90,
    },
    "barabasi": {
        "generator": lambda: _create_barabasi_graphs(),
        "max_prev_node": 130,
    },
    "barabasi_small": {
        "generator": lambda: _create_barabasi_small_graphs(),
        "max_prev_node": 20,
    },
    "ladder": {
        "generator": lambda: _create_ladder_graphs(),
        "max_prev_node": 10,
    },
    "ladder_small": {
        "generator": lambda: _create_ladder_small_graphs(),
        "max_prev_node": 10,
    },
    "caveman": {
        "generator": lambda: _create_caveman_graphs(),
        "max_prev_node": 100,
    },
    "caveman_small": {
        "generator": lambda: _create_caveman_small_graphs(),
        "max_prev_node": 20,
    },
    "caveman_small_single": {
        "generator": lambda: _create_caveman_small_single_graphs(),
        "max_prev_node": 20,
    },
}


# ============================================================================
# Graph Generation Functions (Synthetic Datasets)
# ============================================================================


def _create_grid_graphs() -> list[nx.Graph]:
    """Create standard grid graphs (10x10 to 19x19)."""
    graphs = []
    for i in range(10, 20):
        for j in range(10, 20):
            graphs.append(nx.grid_2d_graph(i, j))
    return graphs


def _create_grid_small_graphs() -> list[nx.Graph]:
    """Create small grid graphs (2x2 to 5x6)."""
    graphs = []
    for i in range(2, 5):
        for j in range(2, 6):
            graphs.append(nx.grid_2d_graph(i, j))
    return graphs


def _create_grid_big_graphs() -> list[nx.Graph]:
    """Create large grid graphs (36x36 to 45x45)."""
    graphs = []
    for i in range(36, 46):
        for j in range(36, 46):
            graphs.append(nx.grid_2d_graph(i, j))
    return graphs


def _create_barabasi_graphs() -> list[nx.Graph]:
    """Create Barabási-Albert graphs (100-199 nodes, degree 4, 5 samples each)."""
    graphs = []
    for i in range(100, 200):
        for j in range(4, 5):
            for k in range(5):
                graphs.append(nx.barabasi_albert_graph(i, j))
    return graphs


def _create_barabasi_small_graphs() -> list[nx.Graph]:
    """Create small Barabási-Albert graphs (4-20 nodes, degree 3, 10 samples each)."""
    graphs = []
    for i in range(4, 21):
        for j in range(3, 4):
            for k in range(10):
                graphs.append(nx.barabasi_albert_graph(i, j))
    return graphs


def _create_ladder_graphs() -> list[nx.Graph]:
    """Create ladder graphs (100-200 nodes)."""
    graphs = []
    for i in range(100, 201):
        graphs.append(nx.ladder_graph(i))
    return graphs


def _create_ladder_small_graphs() -> list[nx.Graph]:
    """Create small ladder graphs (2-10 nodes)."""
    graphs = []
    for i in range(2, 11):
        graphs.append(nx.ladder_graph(i))
    return graphs


def _caveman_special(num_communities: int, size_per_community: int, p_edge: float = 0.5) -> nx.Graph:
    """Generate a caveman graph with specified community structure."""
    graphs = []
    for _ in range(num_communities):
        subgraph = nx.complete_graph(size_per_community)
        graphs.append(subgraph)
    G = nx.disjoint_union_all(graphs)
    
    # Add inter-community edges
    communities = [list(range(i * size_per_community, (i + 1) * size_per_community)) for i in range(num_communities)]
    for i in range(num_communities):
        for j in range(i + 1, num_communities):
            for u in communities[i]:
                for v in communities[j]:
                    if random.random() < p_edge:
                        G.add_edge(u, v)
    return G


def _create_caveman_graphs() -> list[nx.Graph]:
    """Create caveman graphs (2 communities, 30-80 nodes each, 10 samples each)."""
    graphs = []
    for i in range(2, 3):
        for j in range(30, 81):
            for k in range(10):
                graphs.append(_caveman_special(i, j, p_edge=0.3))
    return graphs


def _create_caveman_small_graphs() -> list[nx.Graph]:
    """Create small caveman graphs (2 communities, 6-10 nodes each, 20 samples each)."""
    graphs = []
    for i in range(2, 3):
        for j in range(6, 11):
            for k in range(20):
                graphs.append(_caveman_special(i, j, p_edge=0.8))
    return graphs


def _create_caveman_small_single_graphs() -> list[nx.Graph]:
    """Create single caveman graphs (2 communities, 8 nodes each, 100 samples)."""
    graphs = []
    for i in range(2, 3):
        for j in range(8, 9):
            for k in range(100):
                graphs.append(_caveman_special(i, j, p_edge=0.5))
    return graphs


# ============================================================================
# Public Functions
# ============================================================================


def create_default_graphs(graph_type: str) -> tuple[list[nx.Graph], int]:
    """
    Create graphs for the specified dataset type.
    
    Args:
        graph_type: Type of dataset (e.g., 'grid', 'barabasi_small')
        
    Returns:
        Tuple of (graphs, max_prev_node) where graphs is a list of networkx Graph objects
        and max_prev_node is the max previous node parameter for this dataset.
        
    Raises:
        ValueError: If graph_type is not supported.
    """
    if graph_type not in DATASET_CONFIGS:
        supported = ", ".join(sorted(DATASET_CONFIGS.keys()))
        raise ValueError(
            f"Graph type '{graph_type}' is not supported. "
            f"Supported types: {supported}"
        )
    
    config = DATASET_CONFIGS[graph_type]
    graphs = config["generator"]()
    max_prev_node = config["max_prev_node"]
    
    return graphs, max_prev_node


def split_graphs(
    graphs: list[nx.Graph], 
    seed: int = 123,
    train_ratio: float = 0.8,
    validate_ratio: float = 0.2,
) -> tuple[list[nx.Graph], list[nx.Graph], list[nx.Graph]]:
    """
    Split graphs into train, validate, and test sets.
    
    Args:
        graphs: List of networkx Graph objects
        seed: Random seed for reproducibility
        train_ratio: Fraction of graphs to use for training
        validate_ratio: Fraction of graphs to use for validation (from training set)
        
    Returns:
        Tuple of (train_graphs, validate_graphs, test_graphs)
    """
    random.seed(seed)
    graphs_copy = list(graphs)
    random.shuffle(graphs_copy)
    total = len(graphs_copy)

    test_start_idx = int(train_ratio * total)
    train_graphs = graphs_copy[:test_start_idx]
    test_graphs = graphs_copy[test_start_idx:]
    
    # Validate is subset of training
    validate_end_idx = int(validate_ratio * len(train_graphs))
    validate_graphs = train_graphs[:validate_end_idx]
    
    return train_graphs, validate_graphs, test_graphs


def get_supported_datasets() -> list[str]:
    """Return a list of all supported dataset types."""
    return sorted(DATASET_CONFIGS.keys())
