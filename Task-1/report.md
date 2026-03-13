# Task 1 Report: Convolutional-Recurrent Architectures for WikiArt Multi-Task Classification

## 1. Objective
This task builds a model to classify each painting on three attributes:
- `style`
- `genre`
- `artist`

It also identifies likely outliers, i.e., paintings whose assigned labels may not match visual content, using model confidence and prediction errors.

## 2. Approach (General and Specific)
### General approach
A shared visual backbone is used to learn common features across all tasks, followed by task-specific prediction heads. This is appropriate because style, genre, and artist are correlated visual concepts and can benefit from shared representation learning.

### Specific approach implemented
Two models were trained and compared:
- **Baseline CNN**: EfficientNet-B0 shared backbone with multi-task heads.
- **Convolutional-Recurrent model**: EfficientNet-B0 feature extractor + recurrent sequence modeling over spatial feature tokens (GRU) + multi-task heads.

Why the conv-recurrent model is appropriate:
- CNN captures local texture and composition cues.
- Recurrent modeling over feature tokens can encode longer-range structural dependencies (layout, repeated motifs, brushstroke context), which can help style/genre/artist separation.

## 3. Data and Split Setup
- Dataset root: `data/wikiart/`
- Manifests: `data/manifests/`
- Multi-task training/evaluation uses shared image intersection across `style`, `genre`, `artist`.
- Validation samples scored: **4707**.

## 4. Evaluation Metrics and Why They Were Used
For each task (`style`, `genre`, `artist`), evaluation uses:
- **Top-1 accuracy**: strict correctness for final class prediction.
- **Top-5 accuracy**: useful for fine-grained classes where near-miss classes are semantically close.

For model selection during training:
- **Mean Top-1** across the three tasks was used as the primary checkpoint criterion.

For outlier analysis:
- Per-image prediction exports were analyzed to produce:
  - `high_confidence_errors.csv`: wrong predictions with high confidence (strong outlier/mislabel candidates).
  - `low_confidence_samples.csv`: uncertain samples (ambiguity candidates).

## 5. Quantitative Results (Best Epoch)
Results below are from the best epoch found in each run (both peaked at epoch 5).

| Model | Mean Top-1 | Style Top-1 / Top-5 | Genre Top-1 / Top-5 | Artist Top-1 / Top-5 |
|---|---:|---:|---:|---:|
| Baseline EffB0 | 0.7727 | 0.7674 / 0.9843 | 0.7410 / 0.9896 | 0.8099 / 0.9681 |
| Conv-Recurrent EffB0 | **0.8076** | **0.8171** / **0.9845** | **0.7712** / 0.9887 | **0.8345** / **0.9711** |

Key observation:
- The conv-recurrent variant improved mean Top-1 by about **+3.48 points** (0.8076 vs 0.7727), with strongest gains in style and genre Top-1.

## 6. Outlier Analysis Findings
### Overall error burden (at least one task wrong)
- Baseline: **2178 / 4707**
- Conv-recurrent: **1901 / 4707**
- Reduction: **277 fewer** error samples (**12.72% reduction**).

### Per-task error count changes
- `artist`: 895 -> 779 (**-116**)
- `genre`: 1189 -> 1077 (**-112**)
- `style`: 1081 -> 861 (**-220**)

Interpretation:
- Most high-confidence errors are concentrated in visually neighboring categories (especially genre boundary cases such as portrait vs genre/religious painting).
- Conv-recurrent modeling reduces confusion, especially on style-heavy ambiguity where global composition context matters.

## 7. Representative High-Confidence Outlier Examples
Examples from generated outlier files show typical failure modes:
- **Genre confusion despite correct style/artist**
  - `Impressionism/edgar-degas_the-mante-family.jpg`
  - Predicted `genre_painting` instead of `portrait` with high confidence.
- **Style + artist confusion in stylistically overlapping works**
  - `Expressionism/pablo-picasso_untitled-1958-1.jpg`
  - Style drift to nearby movement and artist confusion among highly expressive painters.
- **Genre boundary ambiguity in symbolic/religious scenes**
  - `Naive_Art_Primitivism/marc-chagall_david-and-bathsheba-1956-1.jpg`
  - Predicted `illustration` instead of `religious_painting`.


Final result: **the conv-recurrent model is the stronger approach on this setup**, with higher mean Top-1 and fewer high-confidence error/outlier candidates.
