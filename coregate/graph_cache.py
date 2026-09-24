"""Small on-disk cache for the expensive JSON-to-tensor graph conversion."""
from pathlib import Path

import torch

from linux_graph_loader import graph_tensor


def load_cached_graphs(catalog_path, rows, graph_field):
    """Return graph tensors in ``rows`` order, caching by sample id/path.

    Graph construction is deterministic and dominates repeated LOEO ablations;
    the cache only stores parsed tensors and never changes the split or labels.
    """
    catalog_path = Path(catalog_path)
    cache_path = catalog_path.parent / (
        f".{catalog_path.stem}_{graph_field}_tensor_cache.pt")
    keys = [(row.get("sample_id"), str(row[graph_field])) for row in rows]
    if cache_path.exists():
        try:
            cached = torch.load(cache_path, map_location="cpu")
            if cached.get("keys") == keys:
                return cached["graphs"]
        except (OSError, RuntimeError, ValueError, KeyError):
            pass
    graphs = [graph_tensor(row[graph_field]) for row in rows]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"keys": keys, "graphs": graphs}, cache_path)
    return graphs
