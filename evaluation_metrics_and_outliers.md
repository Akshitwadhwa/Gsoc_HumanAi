# Evaluation Metrics and Outlier Analysis (Task 1 + Task 2)

## Task 1: Style/Genre/Artist Classification

### Metrics used to evaluate performance
Task 1 is a multi-task classification setup (`style`, `genre`, `artist`), so the following metrics are used per task:

- **Top-1 Accuracy**: strict correctness of the highest-probability class.
- **Top-5 Accuracy**: useful for fine-grained art classes where nearby classes are visually similar.
- **Mean Top-1 (across tasks)**: model-selection metric during training.

Why these are appropriate:
- Top-1 captures final classification quality.
- Top-5 captures ranking quality under label overlap/visual ambiguity.
- Mean Top-1 gives a balanced summary across all three tasks.

### Best validation results (epoch 5)
- **Baseline EffB0**
  - Mean Top-1: **0.7727**
  - Style: Top-1 **0.7674**, Top-5 **0.9843**
  - Genre: Top-1 **0.7410**, Top-5 **0.9896**
  - Artist: Top-1 **0.8099**, Top-5 **0.9681**

- **Conv-Recurrent EffB0**
  - Mean Top-1: **0.8076**
  - Style: Top-1 **0.8171**, Top-5 **0.9845**
  - Genre: Top-1 **0.7712**, Top-5 **0.9887**
  - Artist: Top-1 **0.8345**, Top-5 **0.9711**

### Outlier method (Task 1)
Outliers are found from prediction exports using:
- `high_confidence_errors.csv`: wrong predictions with very high confidence (strong mislabel/outlier candidates).
- `low_confidence_samples.csv`: low-confidence samples (ambiguous/boundary cases).

### Standout outliers (Task 1)
From top high-confidence errors:

1. `Naive_Art_Primitivism/marc-chagall_under-the-snow-1964.jpg`
- Style correct, artist correct, **genre incorrect** (`genre_painting` -> `religious_painting`) at very high confidence.

2. `Impressionism/pierre-auguste-renoir_madame-stora-in-algerian-dress-1870.jpg`
- **Style + artist both incorrect** (`Impressionism` -> `Realism`, `Pierre_Auguste_Renoir` -> `John_Singer_Sargent`) with high confidence.

3. `Expressionism/pablo-picasso_untitled-1958-1.jpg`
- In conv-recurrent outliers: **style + artist incorrect** (`Expressionism` -> `Post_Impressionism`, `Pablo_Picasso` -> `Vincent_van_Gogh`) while genre remains correct.

### Outlier summary impact
- Baseline samples with >=1 wrong task: **2178 / 4707**
- Conv-recurrent samples with >=1 wrong task: **1901 / 4707**
- Reduction: **277 fewer** (about **12.72%**)

---

## Task 2: Retrieval by Artist Similarity

### Metrics used to evaluate performance
Task 2 is retrieval, so ranking-based metrics are used:

- **Precision@10**: fraction of top-10 retrieved items that match query artist.
- **Recall@10**: fraction of all relevant items recovered in top-10.
- **MAP@10**: ranking quality across all relevant hits in top-10.
- **MRR**: how early the first relevant result appears.
- **Queries evaluated**: evaluation coverage count.

Why these are appropriate:
- Retrieval quality depends on rank ordering, not only classification correctness.
- MAP and MRR capture ranking behavior better than plain accuracy.

### Task 2 metric results
From current output folders:

- **ResNet run** (`Task-2/outputs/evaluation_artist_name_top10.csv`)
  - Precision@10: **0.112981**
  - Recall@10: **0.332479**
  - MAP@10: **0.244879**
  - MRR: **0.377223**
  - Queries evaluated: **208**

- **ViT-B/16 baseline** (`Task-2/outputs/vit_b16_baseline/evaluation_artist_name_top10.csv`)
  - Precision@10: **0.101923**
  - Recall@10: **0.320509**
  - MAP@10: **0.215778**
  - MRR: **0.340635**
  - Queries evaluated: **208**

- **EfficientNet-B0 baseline** (`Task-2/outputs/efficientnet_b0_baseline/evaluation_artist_name_top10.csv`)
  - Precision@10: **0.112981**
  - Recall@10: **0.347756**
  - MAP@10: **0.239921**
  - MRR: **0.384581**
  - Queries evaluated: **208**

### Outlier method (Task 2)
Task 2 outliers are treated as **retrieval-mismatch outliers**:
- Queries where top results are from different artists than the query artist.
- Particularly strong cases: **0/6 same-artist hits** in saved query results.

### Standout outliers (Task 2)
Worst query cases (from `multiple_queries` outputs):

1. Query `72` (artist: `Holbein the Younger, Hans`)
- ResNet top result artist: `Botticelli, Sandro`
- ViT top result artist: `Rembrandt van Rijn`
- Same-artist hits in top-6: **0**

2. Query `86` (artist: `Dutch 17th Century`)
- ResNet top result artist: `Benvenuto di Giovanni`
- ViT top result artist: `Gainsborough, Thomas`
- Same-artist hits in top-6: **0**

3. Query `1155` (artist: `David, Gerard`)
- ResNet top result artist: `Dyck, Anthony van, Sir`
- ViT top result artist: `Neroccio de' Landi`
- Same-artist hits in top-6: **0**

Additional summary from saved 10-query stress set:
- Average same-artist precision@6: **0.15** for both ResNet and ViT sets.
- Top-1 artist mismatch queries: **7/10** (ResNet), **8/10** (ViT).

---

## Final Note
- **Task 1** metrics + outlier pipeline are complete and show clear gains from conv-recurrent modeling.
- **Task 2** uses correct retrieval metrics and highlights hard query outliers where artist similarity retrieval still fails.
