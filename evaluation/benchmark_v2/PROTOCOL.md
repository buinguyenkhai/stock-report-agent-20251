# Benchmark v2 Protocol

## Scope
- Evaluate `balance_sheet`, `income_statement`, `cash_flow` tables only.
- Exclude notes (`thuyet minh`) from core benchmark v1.
- Report two layers:
  - Raw OCR table-only fidelity
  - End-to-end structured output fidelity (report-level)

## Split Policy
- Required splits: `dev` and `test`.
- Company-heldout constraint:
  - A company must appear in exactly one split.
  - `dev` is used for threshold/hyperparameter selection.
  - `test` is used once for final reporting.
- Pilot-only exception:
  - For early local/Kaggle smoke tests on partially annotated data, you may run a single split such as `dev`.
  - Do not present single-split results as final benchmark numbers.

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
- `gt_csv/<sample_id>/cells.csv`
- `gt_csv/<sample_id>/rows.csv`
- optional: `gt_csv/<sample_id>/meta.json`
- `page_image_path`
- `gt_markdown_path`
- `gt_structured_path`
- Optional: `gt_table_cells_path`
- Optional: `source_pdf_path`

Canonical numeric contract:
- `gt_csv/<sample_id>/rows.csv:value` and `gt_structured/<sample_id>.json` values must be stored in `VND`.
- Raw OCR references (`gt_markdown`, `gt_cells`) should still preserve the page text exactly as printed, including original page units.

If `source_pdf_path` is present, you can auto-render `page_image_path` for all samples:
```bash
python -m evaluation.benchmark_v2.render_page_images \
  --dataset-root data/benchmark_v2 \
  --split all \
  --dpi 200
```

CSV-first annotation app:
```bash
streamlit run evaluation/benchmark_v2/annotation_app.py
```

Include registry:
- `included_samples.json` defines the small curated subset of fully annotated pages.
- Benchmark CLIs support `--include-scope included` to evaluate only that subset.
- Any sample listed in `included_samples.json` should have complete `gt_csv`, `gt_markdown`, and `gt_structured` artifacts.

## Prediction File Convention
- `<predictions_root>/<sample_id>.raw.md`
- `<predictions_root>/<sample_id>.structured.json`
- generated report-level structured:
  - `<predictions_root>/report_structured/<report_id>.structured.json`

## Raw Scoring Policy
- Default raw scope is `table_only`.
- Structured scoring is always `report-level`:
  - keep `gt_structured/<sample_id>.json` per page for annotation/debug
  - benchmark assembles all pages in the same `report_id` (ordered by `page_index`) before scoring
- CLI:
```bash
python -m evaluation.benchmark_v2.run \
  --dataset-root data/benchmark_v2 \
  --predictions-root results/hybrid_predictions \
  --split test \
  --raw-scope table_only \
  --output results/benchmark_v2_hybrid_test.json
```

## Running Evaluation
Generate predictions first:
```bash
python -m evaluation.benchmark_v2.predict \
  --dataset-root data/benchmark_v2 \
  --output-root results/hybrid_predictions \
  --engine hybrid \
  --split test \
  --include-scope included \
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
  --include-scope included \
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
