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
- `images/<sample_id>.png` (or equivalent path in manifest)
- `gt_markdown/<sample_id>.md`
- `gt_structured/<sample_id>.json`
- optional: `gt_cells/<sample_id>.json`

Manifest fields to fill:
- `annotator_id`: your ID
- `annotation_passes`: set to `2`
- `audited_by`: optional advisor/peer for audited subset

## 3. Labeling order (important)
Use this order for consistency:
1. Read PDF page carefully (source of truth).
2. Build canonical table markdown (`gt_markdown`).
3. Build structured JSON (`gt_structured`) from that page:
   - `balance_sheet.items[]`
   - `income_statement.items[]`
   - `cash_flow.items[]`
4. Normalize numeric values to numeric type in JSON.

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

