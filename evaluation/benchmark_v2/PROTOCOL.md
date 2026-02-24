# Benchmark v2 Protocol

## Scope
- Evaluate `balance_sheet`, `income_statement`, `cash_flow` tables only.
- Exclude notes (`thuyet minh`) from core benchmark v1.
- Report two layers:
  - Raw OCR markdown/table fidelity
  - End-to-end structured output fidelity

## Split Policy
- Required splits: `dev` and `test`.
- Company-heldout constraint:
  - A company must appear in exactly one split.
  - `dev` is used for threshold/hyperparameter selection.
  - `test` is used once for final reporting.

## Single-Annotator Policy
- Minimum QA process for each sample:
  - Pass 1 annotation from PDF page
  - Pass 2 self-review after a time gap
- Manifest should record:
  - `annotator_id`
  - `annotation_passes`
  - `audited_by` (if available)

Detailed annotation guide:
- `evaluation/benchmark_v2/ANNOTATION_GUIDE.md`

## Dataset Artifacts Per Sample
- `page_image_path`
- `gt_markdown_path`
- `gt_structured_path`
- Optional: `gt_table_cells_path`
- Optional: `source_pdf_path`

## Prediction File Convention
- `<predictions_root>/<sample_id>.raw.md`
- `<predictions_root>/<sample_id>.structured.json`

## Running Evaluation
Generate predictions first:
```bash
python -m evaluation.benchmark_v2.predict \
  --dataset-root data/benchmark_v2 \
  --output-root results/hybrid_predictions \
  --engine hybrid \
  --split test \
  --device cuda \
  --hybrid-threshold 0.90 \
  --hybrid-number-threshold 0.95
```

Then score:
```bash
python -m evaluation.benchmark_v2.run \
  --dataset-root data/benchmark_v2 \
  --predictions-root results/hybrid_predictions \
  --engine-name hybrid_docling \
  --split test \
  --output results/benchmark_v2_hybrid_test.json
```

## Hyperparameters and Tuning
Available prediction knobs:
- `--hybrid-threshold`
- `--hybrid-number-threshold`
- `--hybrid-options-json` (extra `HybridOcrOptions` overrides)
- `--device` (`cuda`/`cpu`)
- Marker-specific:
  - `--marker-use-llm`
  - `--marker-llm-model`
  - `--marker-no-force-ocr`
  - `--marker-extract-images`

Tune on dev split only:
```bash
python -m evaluation.benchmark_v2.tune_hybrid \
  --dataset-root data/benchmark_v2 \
  --work-root results/tuning_hybrid \
  --device cuda \
  --hybrid-thresholds 0.70,0.80,0.90 \
  --hybrid-number-thresholds 0.85,0.90,0.95 \
  --objective blended
```

Use best thresholds from `results/tuning_hybrid/tuning_summary.json`, then run final test once.
