# Task 2 Report: Painting Similarity

## Problem framing

Task 2 is best treated as an image-retrieval problem rather than a classification problem.  
My goal was not to assign a single label, but to return paintings that are visually similar to a query image, especially for portrait-like cases where similarity may come from:

- face structure
- pose
- composition
- color treatment
- overall visual style

## Dataset

This baseline uses the National Gallery of Art open-access dataset.

Current prepared data:

- portrait-focused prepared metadata rows: `431`
- indexed rows with downloaded local images: `300`

The indexed set is almost entirely `Painting` entries (`295`) with a small number of `Drawing` entries (`5`).

## Method

The baseline pipeline is:

1. Download and merge NGA metadata.
2. Filter to paintings and portrait-oriented examples.
3. Download local images.
4. Extract image embeddings using a pretrained `torchvision` `ResNet50` encoder.
5. L2-normalize embeddings.
6. Compute cosine similarity between the query embedding and all indexed paintings.
7. Return the top-k nearest neighbors.

Why this is a reasonable first baseline:

- it is simple and stable
- it avoids training a model from scratch
- it gives a usable retrieval system quickly
- it provides a clean baseline for later comparison with stronger encoders such as CLIP or DINO

## Evaluation metrics

The retrieval system was evaluated with `artist_name` as the relevance label and `top_k = 10`.

Current results:

- `precision@10 = 0.112981`
- `recall@10 = 0.332479`
- `map@10 = 0.244879`
- `mrr = 0.377223`
- `queries_evaluated = 208`

Interpretation:

- `precision@10` is modest, so many top-10 results are not exact artist matches.
- `recall@10` is better, which means relevant results are often present somewhere in the retrieved list.
- `MAP@10` and `MRR` show that relevant results are often ranked fairly early, but not consistently at the very top.

This is acceptable for a first visual-similarity baseline, but it is not yet strong enough to claim robust portrait-level semantic similarity.

### New encoder experiment: EfficientNet-B0 (offline-friendly)

CLIP/DINO weights could not be downloaded in the offline environment, so I tried the strongest cached pretrained weights available locally: `efficientnet_b0`.

Results (artist_name, top_k = 10):

- `precision@10 = 0.112981`
- `recall@10 = 0.347756`
- `map@10 = 0.239921`
- `mrr = 0.384581`
- `queries_evaluated = 208`

Files: `Task-2/outputs/efficientnet_b0_baseline/embeddings.pt`, `index_metadata.csv`, and `evaluation_artist_name_top10.csv`.

Comparison to the ResNet50 baseline:

- precision unchanged, recall +0.015, MRR +0.007 — a small but consistent gain despite the lightweight model.
- ViT-B/16 remained slightly weaker than both (see `Task-2/outputs/vit_b16_baseline/`).

Next encoder step (once network access is available): swap in CLIP or DINOv2 weights and rerun the same pipeline for a larger jump in portrait-quality retrieval.

## Qualitative examples

### Stronger example: query `565`

Query painting:

- `565` — `George Washington (Vaughan-Sinclair portrait)` by `Stuart, Gilbert`

Top retrieved results include:

- `1121` — `George Washington (Vaughan portrait)` by `Stuart, Gilbert`
- `1119` — `William Thornton` by `Stuart, Gilbert`
- `1114` — `Stephen Van Rensselaer III` by `Stuart, Gilbert`

Observation:

- this is a strong retrieval case
- the system retrieves highly similar formal portraits
- the nearest neighbors share portrait framing, pose conventions, and similar painterly treatment
- the top result is especially convincing because it is another George Washington portrait by the same artist

### Composition-focused example: query `1174`

Query painting:

- `1174` — `The Hoppner Children` by `Hoppner, John`

Top retrieved results include:

- `995` — `The Binning Children`
- `103` — `Lady Mary Templetown and Her Eldest Son`
- `102` — `Lady Elizabeth Delmé and Her Children`

Observation:

- this is a good composition-level retrieval example
- the model appears to respond to grouped figures, portrait staging, and family arrangement
- the results are visually plausible even when the artist changes

### Weaker example: query `63`

Query painting:

- `63` — `A Dutch Courtyard` by `Hooch, Pieter de`

Top retrieved results include:

- `1173` — `Woman and Child in a Courtyard`
- `96` — `La Camargo Dancing`
- `64` — `The Intruder`
- `20` — `The Adoration of the Child`

Observation:

- the first result is strong because it shares a similar domestic courtyard/interior-like setting
- lower-ranked results become less semantically consistent
- retrieval seems driven more by broad composition and scene layout than by a precise notion of subject similarity

### Mixed example: query `426`

Query painting:

- `426` — `Saint John the Baptist` by `Sellaio, Jacopo del`

Top retrieved results include:

- `370` — `Tobias and the Angel`
- `20` — `The Adoration of the Child`
- `31` — `The Crucifixion with the Virgin, Saint John, Saint Jerome, and Saint Mary Magdalene [right panel]`

Observation:

- the first few results stay within a broadly compatible Renaissance religious/figure space
- this suggests the baseline captures period/style/composition cues reasonably well
- however, the retrieval is still broad rather than tightly portrait- or identity-focused

## Overall qualitative conclusion

The current retrieval system appears to capture:

- broad portrait conventions
- composition
- figure arrangement
- painterly style
- some artist-specific visual signatures

It is weaker at:

- fine-grained face similarity
- distinguishing subject identity from broad stylistic similarity
- consistently ranking the most semantically relevant result first

## Limitations

- `ResNet50` features are a general visual baseline, not a retrieval model specialized for fine-grained similarity.
- Evaluation by `artist_name` is useful but incomplete, because visually similar paintings can be relevant even across different artists.
- Portrait similarity in the task statement may require stronger features for face, pose, or human layout.

## Next improvements

Recommended next upgrade path:

1. Replace `ResNet50` embeddings with CLIP or DINO embeddings.
2. Compare the new metrics against this baseline.
3. Keep the same retrieval notebooks and saved query CSV workflow for side-by-side qualitative comparison.
4. If portrait specificity remains weak, add portrait-only features such as face embeddings or pose/keypoint cues.
