# GSoC HumanAI Pre-Project

This repository is set up for the WikiArt convolutional-recurrent pre-project.

- `outputs/`: generated summaries and plots

```bash
python3 scripts/inspect_wikiart.py --dataset-root data --output-dir outputs/day1
```

The script scans CSV label files under the dataset root and writes:

- `dataset_summary.md`
- `split_summary.csv`
- `class_balance.csv`

4. Train the baseline multi-task CNN:

```bash
python3 -m src.train \
  --image-root data/wikiart \
  --manifest-root data/manifests \
  --tasks style genre artist \
  --backbone resnet50 \
  --epochs 10 \
  --batch-size 32 \
  --output-dir outputs/baseline_resnet50
```

5. Evaluate a saved checkpoint:

```bash
python3 -m src.evaluate \
  --checkpoint outputs/baseline_resnet50/best.pt \
  --image-root data/wikiart \
  --manifest-root data/manifests \
  --split val
```

6. Export per-image predictions for error analysis:

```bash
python3 -m src.predict \
  --checkpoint outputs/baseline_resnet50/best.pt \
  --image-root data/wikiart \
  --manifest-root data/manifests \
  --split val \
  --output-csv outputs/baseline_resnet50/val_predictions.csv
```

7. Rank likely outliers from the prediction export:

```bash
python3 scripts/find_outliers.py \
  --predictions-csv outputs/baseline_resnet50/val_predictions.csv \
  --output-dir outputs/baseline_resnet50/outliers \
  --top-n 100
```

## What to look for on Day 1

- How many classes exist for `style`, `genre`, and `artist`
- Whether the validation split is representative
- Which tasks are heavily imbalanced
- Whether any referenced image paths are missing

## Day 2

Use the Day 1 summary to choose a baseline:

- single-task CNN for one attribute first, or
- multi-task CNN if the splits are clean and consistent

## Baseline Model

The project now includes a shared-backbone CNN baseline in `src/`:

- `src/dataset.py`: joins `style`, `genre`, and `artist` manifests on shared image paths
- `src/model.py`: `ResNet50` or `EfficientNet` backbone with one head per task
- `src/train.py`: training loop with top-1 and top-5 validation metrics
- `src/evaluate.py`: checkpoint evaluation on train or validation splits
- `src/predict.py`: per-image prediction export for error analysis
- `scripts/find_outliers.py`: ranks high-confidence errors and low-confidence samples
