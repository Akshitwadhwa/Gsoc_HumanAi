# Outlier Summary

- Total samples scored: 4707
- Samples with at least one wrong task: 2178
- Top ranked rows per CSV: 100

## Task Error Counts

- `artist` errors: 895
- `genre` errors: 1189
- `style` errors: 1081

## Interpretation

- `high_confidence_errors.csv` contains images the model classified incorrectly despite being confident. These are the strongest label/outlier candidates.
- `low_confidence_samples.csv` contains images with the lowest average confidence across tasks. These are ambiguity candidates.
