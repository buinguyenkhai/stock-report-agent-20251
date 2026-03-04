# Annotation Guide for Benchmark v2

This guide is for your chosen setup:
- `dev + test` split
- `single annotator` with `two-pass QA`
- scope: BS/IS/CF tables only (no notes)

## 1. Prepare the sample list first
Before labeling any values, create and freeze:
- company list
- report IDs
- page indices
- split assignment (`dev` vs `test`)

Rules:
- Company-heldout split: one company must appear in only one split.
- Do not move samples between splits after tuning starts.

## 2. Required files per sample
For each `sample_id`, store:
- `gt_csv/<sample_id>/cells.csv`
- `gt_csv/<sample_id>/rows.csv`
- optional: `gt_csv/<sample_id>/meta.json`
- `images/<sample_id>.png` (or equivalent path in manifest)
- `gt_markdown/<sample_id>.md`
- `gt_structured/<sample_id>.json`
- optional: `gt_cells/<sample_id>.json`
- optional (generated): `gt_structured_report/<report_id>.json`

CSV schema:
- `cells.csv`: `row_idx,col_idx,text`
- `rows.csv`: `statement,item_code,item_name,value,notes_ref,original_name`

If your manifest includes `source_pdf_path`, generate images automatically:
```bash
python -m evaluation.benchmark_v2.render_page_images \
  --dataset-root data/benchmark_v2 \
  --split all \
  --dpi 200
```

## 2.1 PDF Download Conventions (Important)
- Put raw PDFs under `<dataset_root>/pdf/`.
- File naming convention (recommended): `<COMPANY>_<REPORT_ID>.pdf`  
  Example: `FPT_2024Q3.pdf`, `VCB_2024_annual.pdf`
- `COMPANY` should be the ticker/code at the start of filename because auto-manifest infers company from prefix before first `_`.
- Avoid spaces and special characters in filenames; prefer `[A-Z0-9_]`.
- Keep one report per PDF file (do not concatenate unrelated reports).
- Scanned PDFs are supported (auto-manifest reads page count only; OCR quality is handled later).
- Suggested pilot size: start with 20 pages (mixed companies + statements), then scale.
- No strict maximum file size/page count, but very large PDFs (>300 pages) should use `Max pages per PDF` in app to bootstrap gradually.

Manifest fields to fill:
- `annotator_id`: your ID
- `annotation_passes`: set to `2`
- `audited_by`: optional advisor/peer for audited subset

Run annotation app (CSV-first):
```bash
streamlit run evaluation/benchmark_v2/annotation_app.py
```
If `manifest.json` is missing, the app now provides a setup wizard to create it
(empty template or auto-generated from `pdf/*.pdf`).

GUI-first workflow:
1. Drop PDFs into `<dataset_root>/pdf`.
2. Open app and click `Auto-Generate Manifest from pdf/*.pdf`.
3. In `Dataset Ops`, click `Render Images from Manifest`.
4. Keep `Include Scope = not_included`, then mark table pages via `Include Current Sample (Table)`.
5. Annotate included pages in CSV editors and save canonical outputs.
6. In `Dataset Ops`, click `Build GT Report Structured Files` to generate report-level structured JSON.

## 3. Labeling order (important)
Use this order for consistency:
1. Read PDF page carefully (source of truth).
2. Fill CSV pack in app:
   - `cells.csv`
   - `rows.csv`
3. Generate canonical outputs from CSV:
   - `gt_markdown`
   - `gt_structured`
   - optional `gt_cells`
4. Run pass-2 QA and update correction counters in `meta.json`.

Do not copy from OCR output first. OCR output can be used only as a check after you finish manual labeling.

## 4. Canonical conventions
Apply these consistently:
- Keep statement assignment correct (`BS/IS/CF`).
- Prefer `item_code` when present; fallback to `item_name`.
- `value` should be numeric (`number`), `null` if absent.
- Keep `item_name` faithful to report wording.
- Keep `notes_ref` if visible in the statement row.

Numeric normalization:
- `(1.234)` => negative numeric value
- keep decimals when present
- do not silently round unless source itself is rounded

## 5. Two-pass QA for single annotator
Pass 1:
- annotate all fields

Pass 2 (after a time gap):
- re-open the same page
- verify row identity, statement assignment, and all numbers
- fix errors and increment confidence notes (if you keep local notes)

Recommended audit:
- 10-15% pages reviewed by advisor/peer
- set `audited_by` in manifest for those samples

## 6. Quick quality checklist per page
- Are all intended rows captured?
- Any duplicated rows?
- Any row in wrong statement?
- Any numeric sign errors (`+/-`)?
- Any decimal/thousands separator mistakes?
- Any missing high-impact totals/subtotals?

## 7. Validate dataset before benchmarking
Run prediction/eval only after manifest and files are stable.

Minimal checks:
- both `dev` and `test` exist
- no overlapping company across splits
- referenced files exist

## 8. Recommended annotation throughput
For 200 pages:
- pilot 20 pages first (calibrate conventions)
- then annotate in batches of 20-30 pages
- run QA immediately after each batch

This catches drift early and avoids costly relabeling near the end.
