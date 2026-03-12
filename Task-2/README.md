# Task 2: Painting Similarity

This folder contains a practical starting point for the similarity task using the National Gallery of Art open data.

## Goal

Build an image-retrieval system that returns visually similar paintings, with a portrait-focused workflow for queries like:

- portraits with a similar face
- portraits with a similar pose
- paintings with a similar composition or color treatment

## Strategy

This task is better framed as retrieval instead of classification.

Recommended first approach:

1. collect open-access NGA paintings and metadata
2. filter to paintings, optionally portrait-heavy examples
3. extract image embeddings using a pretrained visual encoder
4. normalize embeddings and compare them with cosine similarity
5. retrieve top-k nearest neighbors for a query painting
6. evaluate retrieval quality with top-k metrics and qualitative examples

## Why this approach

- Similarity search is naturally an embedding problem.
- A pretrained encoder gives a strong baseline without full retraining.
- Cosine similarity is simple, stable, and fast to compute.
- Portrait filtering lets you specialize the search space for face/pose-like queries.

## Current baseline in this folder

This scaffold uses a pretrained `torchvision` `resnet50` encoder because it already works in the project environment.

Future upgrade options:

- CLIP or DINO embeddings
- fine-tuning with contrastive / triplet loss
- face or keypoint features for portrait-only retrieval

## Files

- `prepare_data.py`: download and merge NGA metadata into a clean working CSV
- `download_images.py`: download open-access images from NGA IIIF URLs
- `build_index.py`: extract embeddings and save an index
- `query.py`: retrieve top-k similar paintings
- `evaluate.py`: compute retrieval metrics
- `task2/nga_similarity.py`: shared data / model / metric utilities

## Expected workflow

Run from this folder:

```bash
cd Task-2
```

### 1. Prepare NGA metadata

```bash
../venv/bin/python prepare_data.py --portrait-only --max-rows 1000
```

This will:

- download the NGA CSV metadata into `data/raw/`
- build a merged table in `data/processed/nga_similarity_metadata.csv`

### 2. Download images

```bash
../venv/bin/python download_images.py --limit 500
```

This saves images into `data/images/`.

### 3. Build the embedding index

```bash
../venv/bin/python build_index.py --batch-size 16
```

This creates:

- `outputs/index_metadata.csv`
- `outputs/embeddings.pt`

### 4. Query similar paintings

By object id:

```bash
../venv/bin/python query.py --object-id 17387 --top-k 8
```

By local image path:

```bash
../venv/bin/python query.py --image-path data/images/17387.jpg --top-k 8
```

### 5. Evaluate retrieval

```bash
../venv/bin/python evaluate.py --label-column artist_name --top-k 10
```

You can also evaluate against:

- `classification`
- `subclassification`
- `portrait_flag`

## Evaluation metrics

Use these metrics for retrieval:

- `Precision@K`: how many of the top-k retrieved items are relevant
- `Recall@K`: how many relevant items were recovered in the top-k
- `MAP@K`: ranking quality across the retrieved list
- `MRR`: how early the first relevant result appears

For the report, pair the metrics with qualitative retrieval examples:

- one query image
- its top-k neighbors
- a short explanation of why the retrieval is good or bad

## Recommended report structure

1. problem framing: why similarity is retrieval, not classification
2. dataset preparation: how paintings and portraits are filtered
3. model choice: why pretrained image embeddings are used
4. similarity function: cosine similarity
5. metrics: Precision@K, Recall@K, MAP@K, MRR
6. qualitative examples: nearest-neighbor grids
7. limitations and next improvements

## Practical next steps

1. Run the metadata preparation script.
2. Download a manageable subset first, e.g. 300 to 1000 images.
3. Build embeddings.
4. Evaluate with `artist_name` and `classification`.
5. Inspect portrait queries qualitatively.
6. If the baseline is weak, upgrade the encoder or add portrait-specific features.
