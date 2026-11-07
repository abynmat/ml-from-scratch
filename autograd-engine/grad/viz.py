"""
grad.viz -- draw a grad.Tensor computation graph with networkx + matplotlib
(Author: Abyn Scaria). No graphviz binary required.

Layout: nodes are placed in columns by their depth from the leaves (a
topological "generation"), so data flows left to right and the backward pass
reads right to left.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .tensor import Tensor


def build_graph(root: Tensor) -> nx.DiGraph:
    """Collect every node reachable from ``root`` (call before backward, or with retain_graph)."""
    G = nx.DiGraph()
    stack, seen = [root], set()
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        G.add_node(id(n), t=n)
        for p in n._parents:
            G.add_edge(id(p), id(n))
            stack.append(p)
    return G


def _label(t: Tensor) -> str:
    head = t._op or (t.name or ("leaf" if t.requires_grad else "const"))
    if t._op and t.name:
        head = f"{t.name} = {t._op}"
    shape = "x".join(map(str, t.shape)) or "scalar"
    lines = [head, f"[{shape}]"]
    if t.size <= 2:
        lines.append("val " + ", ".join(f"{v:.3f}" for v in t.data.reshape(-1)))
    if t.grad is not None:
        lines.append(("grad " + ", ".join(f"{v:.3f}" for v in t.grad.reshape(-1)))
                     if t.grad.size <= 2 else f"|grad| {np.linalg.norm(t.grad):.3f}")
    return "\n".join(lines)


def draw(root: Tensor, path: str, title: str = "", figsize=(16, 6.5)):
    G = build_graph(root)
    depth = {}
    for n in nx.topological_sort(G):
        preds = list(G.predecessors(n))
        depth[n] = 0 if not preds else 1 + max(depth[p] for p in preds)
    cols = {}
    for n, d in depth.items():
        cols.setdefault(d, []).append(n)
    pos = {}
    for d, nodes in cols.items():
        for k, n in enumerate(sorted(nodes, key=lambda i: G.nodes[i]["t"].name)):
            pos[n] = (d, -(k - (len(nodes) - 1) / 2))
    colours = ["#cfe8ff" if G.nodes[n]["t"].is_leaf else "#ffe2c2" for n in G.nodes]
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=14, node_size=5200,
                           edge_color="#666")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colours, node_size=5200,
                           node_shape="s", edgecolors="#333")
    nx.draw_networkx_labels(G, pos, {n: _label(G.nodes[n]["t"]) for n in G.nodes},
                            font_size=7, ax=ax)
    ax.set_title(title or "computation graph (blue = leaf, orange = op)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return G
