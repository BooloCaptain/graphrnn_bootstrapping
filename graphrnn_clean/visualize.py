import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def draw_graph_list(
    graph_list,
    row,
    col,
    fname,
    layout="spring",
    is_single=False,
    k=1,
    node_size=55,
    alpha=1,
    width=1.3,
):
    plt.switch_backend("agg")
    for i, graph in enumerate(graph_list):
        plt.subplot(row, col, i + 1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.axis("off")

        if layout == "spring":
            pos = nx.spring_layout(graph, k=k / np.sqrt(graph.number_of_nodes()), iterations=100)
        elif layout == "spectral":
            pos = nx.spectral_layout(graph)
        else:
            pos = nx.spring_layout(graph, k=k / np.sqrt(graph.number_of_nodes()), iterations=100)

        if is_single:
            nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color="#336699", alpha=1, linewidths=0)
            nx.draw_networkx_edges(graph, pos, alpha=alpha, width=width)
        else:
            nx.draw_networkx_nodes(graph, pos, node_size=1.5, node_color="#336699", alpha=1, linewidths=0.2)
            nx.draw_networkx_edges(graph, pos, alpha=0.3, width=0.2)

    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=600)
    plt.close()
