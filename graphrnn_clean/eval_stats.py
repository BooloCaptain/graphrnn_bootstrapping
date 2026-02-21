import concurrent.futures
from datetime import datetime
import os
import subprocess as sp

import networkx as nx
import numpy as np

from . import eval_mmd as mmd

PRINT_TIME = False


def _degree_worker(graph):
    return np.array(nx.degree_histogram(graph))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=False, use_emd=True):
    sample_ref = []
    sample_pred = []

    graph_pred_list_remove_empty = [graph for graph in graph_pred_list if graph.number_of_nodes() > 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(_degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(_degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for graph in graph_ref_list:
            sample_ref.append(np.array(nx.degree_histogram(graph)))
        for graph in graph_pred_list_remove_empty:
            sample_pred.append(np.array(nx.degree_histogram(graph)))

    kernel = mmd.gaussian_emd if use_emd else mmd.gaussian
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=kernel, is_parallel=is_parallel)
    if PRINT_TIME:
        print("Time computing degree mmd:", datetime.now() - prev)
    return mmd_dist


def _clustering_worker(param):
    graph, bins = param
    clustering_coeffs = list(nx.clustering(graph).values())
    hist, _ = np.histogram(clustering_coeffs, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=False, use_emd=True):
    sample_ref = []
    sample_pred = []

    graph_pred_list_remove_empty = [graph for graph in graph_pred_list if graph.number_of_nodes() > 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(_clustering_worker, [(g, bins) for g in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(
                _clustering_worker, [(g, bins) for g in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)
    else:
        for graph in graph_ref_list:
            clustering_coeffs = list(nx.clustering(graph).values())
            hist, _ = np.histogram(clustering_coeffs, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for graph in graph_pred_list_remove_empty:
            clustering_coeffs = list(nx.clustering(graph).values())
            hist, _ = np.histogram(clustering_coeffs, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    if use_emd:
        mmd_dist = mmd.compute_mmd(
            sample_ref, sample_pred, kernel=mmd.gaussian_emd, sigma=1.0 / 10, distance_scaling=bins, is_parallel=is_parallel
        )
    else:
        mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian, sigma=1.0 / 10, is_parallel=is_parallel)
    if PRINT_TIME:
        print("Time computing clustering mmd:", datetime.now() - prev)
    return mmd_dist


_MOTIF_TO_INDICES = {
    "3path": [1, 2],
    "4cycle": [8],
}
_COUNT_START_STR = "orbit counts: \n"


def _edge_list_reindexed(graph):
    idx = 0
    id2idx = {}
    for node in graph.nodes():
        id2idx[str(node)] = idx
        idx += 1

    edges = []
    for u, v in graph.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def _orca(graph):
    tmp_fname = "eval/orca/tmp.txt"
    with open(tmp_fname, "w") as file:
        file.write(f"{graph.number_of_nodes()} {graph.number_of_edges()}\n")
        for u, v in _edge_list_reindexed(graph):
            file.write(f"{u} {v}\n")

    output = sp.check_output(["./eval/orca/orca", "node", "4", "eval/orca/tmp.txt", "std"])
    output = output.decode("utf8").strip()

    idx = output.find(_COUNT_START_STR) + len(_COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array(
        [list(map(int, node_cnts.strip().split(" "))) for node_cnts in output.strip("\n").split("\n")]
    )

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list, is_parallel=False):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [graph for graph in graph_pred_list if graph.number_of_nodes() > 0]

    for graph in graph_ref_list:
        try:
            orbit_counts = _orca(graph)
        except Exception:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / graph.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for graph in graph_pred_list_remove_empty:
        try:
            orbit_counts = _orca(graph)
        except Exception:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / graph.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = mmd.compute_mmd(total_counts_ref, total_counts_pred, kernel=mmd.gaussian, is_hist=False, sigma=30.0, is_parallel=is_parallel)
    return mmd_dist
