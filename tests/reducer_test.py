import sys
import os
import torch

# --- Add project root to path to allow imports from src ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------------------

from models.reducer import PCA


reducer = PCA(
    n_components=5
)

embeddings = [torch.randn(512) for _ in range(500)]

# embeddings = List[torch.Tensor] with shape (512,) for each
reducer.fit(embeddings)

reduced_embeddings = reducer.reduce_dimensions(embeddings)
print(reduced_embeddings)