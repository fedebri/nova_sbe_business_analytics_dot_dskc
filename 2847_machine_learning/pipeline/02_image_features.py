"""Extract frozen ResNet-18 features and within-type image neighbors.

This stage is separate from fitting so CV and the final refit reuse
identical catalog-only features. First execution may download the pretrained
weights through torchvision. Nothing in this file runs during static checks.
"""

import json
import logging
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


INPUT_ARTICLES = Path("data/processed/articles.parquet")
INPUT_IMAGES = Path("data/full/images")
OUTPUT_EMBEDDINGS = Path("results/features/image_embeddings.npz")
OUTPUT_NEIGHBORS = Path("results/features/image_neighbors.npz")
OUTPUT_STATUS = Path("results/features/image_status.parquet")
OUTPUT_MANIFEST = Path("results/features/image_manifest.json")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
articles = pd.read_parquet(INPUT_ARTICLES)
article_ids = articles["article_id"].to_numpy(dtype=str)
# Image subfolders use the first three digits of each ten-digit article ID.
paths = [
    INPUT_IMAGES / article[:3] / f"{article}.jpg" for article in article_ids
]
present = np.array([path.is_file() for path in paths])
# Keep one feature row per catalog article, even when its image is missing.
embeddings = np.zeros((len(articles), 512), dtype=np.float32)
readable = np.zeros(len(articles), dtype=bool)
status = np.where(present, "pending", "missing").astype(object)

# An empty image set needs neither a model nor a weight download.
device_name = "not_used"
preprocessing = "ResNet18_Weights.IMAGENET1K_V1.transforms()"
if present.any():
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device_name = str(device)
    weights = ResNet18_Weights.IMAGENET1K_V1
    # Use the resizing, cropping, and pixel scaling expected by these weights.
    transform = weights.transforms()
    preprocessing = repr(transform)
    encoder = resnet18(weights=weights)
    # Keep 512 visual features per article rather than ImageNet class scores.
    encoder.fc = torch.nn.Identity()
    # Freeze weights and use saved normalization statistics at inference.
    encoder.requires_grad_(False)
    encoder.eval().to(device)
    existing = np.flatnonzero(present)
    # Encode 32 images at a time to limit accelerator memory.
    for start in range(0, len(existing), 32):
        stop = start + 32
        batch_rows = []
        batch_images = []
        for row in existing[start:stop]:
            try:
                with Image.open(paths[row]) as image:
                    # Respect image orientation and provide three RGB channels.
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    batch_images.append(transform(image))
                batch_rows.append(row)
            except (UnidentifiedImageError, OSError) as error:
                # A corrupt catalog image must not prevent metadata fallback.
                status[row] = "unreadable"
                logging.warning("Skipping image %s: %s", paths[row], error)
        if batch_rows:
            with torch.inference_mode():
                features = encoder(torch.stack(batch_images).to(device))
            embeddings[batch_rows] = features.cpu().numpy()
            readable[batch_rows] = True
            status[batch_rows] = "ok"
        if start % 3200 == 0:
            logging.info("Images visited: %s / %s", start, len(existing))
    # Give readable vectors length 1 while preserving their visual direction.
    embeddings[readable] = normalize(embeddings[readable], norm="l2")

# Retain neighbors for the image diagnostic notebook and cache identity.
# The selected ranker uses embeddings as profile features, not these edges.
# Each article has up to 20 neighbor slots; -1 marks an unused slot.
neighbors = np.full((len(articles), 20), -1, dtype=np.int32)
similarities = np.zeros((len(articles), 20), dtype=np.float32)
# Search within product type to find visual substitutes for purchased articles.
for _, positions in articles.groupby(
    "product_type_name", dropna=False
).indices.items():
    # Zero placeholders for missing images must not become visual neighbors.
    positions = np.asarray(positions)[readable[positions]]
    if not len(positions):
        continue
    search = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1)
    search.fit(embeddings[positions])
    # Query in batches to avoid storing all pairwise distances.
    for start in range(0, len(positions), 256):
        stop = start + 256
        query = positions[start:stop]
        # One extra result allows us to remove the article itself.
        distances, local_indices = search.kneighbors(
            embeddings[query], n_neighbors=min(21, len(positions))
        )
        for row, distance, local_index in zip(
            query, distances, local_indices, strict=True
        ):
            # Map positions within this type back to catalog rows.
            destinations = positions[local_index]
            # Remove self-matches and pairs with no positive cosine similarity.
            keep = (destinations != row) & (distance < 1.0)
            destinations, distance = destinations[keep], distance[keep]
            # Keep the closest 20, breaking equal-distance ties by catalog row.
            order = np.lexsort((destinations, distance))[:20]
            neighbors[row, : len(order)] = destinations[order]
            # Convert distance to similarity and keep roundoff within [0, 1].
            similarities[row, : len(order)] = np.clip(
                1.0 - distance[order], 0.0, 1.0
            )

# Save profile vectors and the separate diagnostic neighbor graph.
np.savez_compressed(
    OUTPUT_EMBEDDINGS,
    article_ids=article_ids,
    embeddings=embeddings,
    readable=readable,
)
np.savez_compressed(
    OUTPUT_NEIGHBORS, indices=neighbors, similarities=similarities
)
pd.DataFrame(
    {
        "article_id": article_ids,
        "product_type_name": articles["product_type_name"],
        "image_path": [str(path) for path in paths],
        "status": status,
    }
).to_parquet(OUTPUT_STATUS, index=False)
# Report image coverage separately from the recommendation-quality metrics.
manifest = {
    "weights": "ResNet18_Weights.IMAGENET1K_V1",
    "frozen": True,
    "dimensions": 512,
    "normalization": "L2",
    "preprocessing": preprocessing,
    "neighbor_scope": (
        "same product_type_name; self excluded; positive cosine only"
    ),
    "neighbor_count": 20,
    "device": device_name,
    "catalog_articles": len(articles),
    "readable_images": int(readable.sum()),
    "status_counts": pd.Series(status).value_counts().astype(int).to_dict(),
    "torchvision_version": version("torchvision") if present.any() else None,
    "note": "Regenerate these artifacts when source image files change.",
}
OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
