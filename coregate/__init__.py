"""CoreGate reference implementation.

Exports the modules used by the released experiment runners:

- ``graph_encoder``: full-graph transformer encoder (edge-order features).
- ``tgn_episode``: episodic prototypical training, pooling, and embedding.
- ``label_blind_lineage``: execution-lineage evidence masks (label-blind target).

``graph_cache`` is intentionally not imported here because it depends on
the dataset loader module; import it directly if you need the tensor cache.
"""

from .graph_encoder import (
    OrderEncoder,
    GraphTransformerEncoder,
)
from .tgn_episode import (
    temporal_graph_forward,
    episode_split,
    episode_prototypes,
    train_episode_tgn,
    embed_episode_graphs,
)
from .label_blind_lineage import build_masks, load_mask_records

__all__ = [
    "OrderEncoder",
    "GraphTransformerEncoder",
    "temporal_graph_forward",
    "episode_split",
    "episode_prototypes",
    "train_episode_tgn",
    "embed_episode_graphs",
    "build_masks",
    "load_mask_records",
]
