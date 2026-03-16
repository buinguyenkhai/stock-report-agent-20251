# Benchmark v2 Protocol

## Scope
- Evaluate OCR quality for financial tables only.
- Core outputs:
  - per-page raw markdown
  - raw OCR metrics
  - OCR debug metadata
  - latency and VRAM telemetry

## Split Policy
- Required splits: `dev` and `test`.
- Company-heldout constraint:
  - a company appears in exactly one split
  - `dev` is for tuning
  - `test` is for final reporting
- Pilot exception:
  - single-split runs such as `dev` are allowed for smoke tests
  - do not report them as final benchmark numbers

## Dataset Artifacts Per Sample
- `page_image_path`
- `gt_markdown_path`
- optional `source_pdf_path`
- optional annotation metadata:
  - `annotator_id`
  - `annotation_passes`
  - `audited_by`
  - `notes`

If `source_pdf_path` is present, render missing page images with:
```bash
python -m evaluation.benchmark_v2.render_page_images \
  --dataset-root data/benchmark_v2 \
  --split all \
  --dpi 200
```

Include registry:
- `included_samples.json` defines the curated subset used for stable OCR comparisons.

## Prediction File Convention
- `<predictions_root>/<sample_id>.raw.md`
- `<predictions_root>/<sample_id>.ocr_debug.json`

`ocr_debug.json` should contain:
- OCR engine debug payload
- reconstruction debug payload
- telemetry:
  - `total_latency_ms`
  - `peak_vram_reserved_mb`
  - `peak_vram_allocated_mb`

## Raw Scoring Policy
- Raw scope is `table_only`.
- Metrics:
  - `table_only_cer`
  - `table_only_wer`
  - `table_cell_f1`
  - `number_f1`

## Running Evaluation
Generate predictions:
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

Build raw diffs for inspection:
```bash
python -m evaluation.benchmark_v2.debug_diffs \
  --dataset-root data/benchmark_v2 \
  --predictions-root results/hybrid_predictions \
  --split test \
  --include-scope included \
  --output results/benchmark_v2_hybrid_test_debug_diffs.json
```

View diffs:
```bash
streamlit run evaluation/benchmark_v2/debug_app.py -- --diff-json results/benchmark_v2_hybrid_test_debug_diffs.json
```

## Hyperparameters and Tuning
Available prediction knobs:
- `--hybrid-threshold`
- `--hybrid-number-threshold`
- `--hybrid-options-json`
- `--device`
- marker-specific:
  - `--marker-use-llm`
  - `--marker-llm-model`
  - `--marker-no-force-ocr`
  - `--marker-extract-images`

Tune on `dev` only:
```bash
python -m evaluation.benchmark_v2.tune_hybrid \
  --dataset-root data/benchmark_v2 \
  --work-root results/tuning_hybrid \
  --device cuda \
  --hybrid-thresholds 0.70,0.80,0.90 \
  --hybrid-number-thresholds 0.85,0.90,0.95 \
  --objective blended_raw
```
