from evaluation.ocr_benchmark.page_level_benchmark import rewrite_docling_table_grid_text_from_ocr_cells


def test_rewrite_docling_table_grid_text_from_ocr_cells_updates_text() -> None:
    export_dict = {
        "tables": [
            {
                "data": {
                    "grid": [
                        [
                            {"text": "OLD", "bbox": {"l": 0, "t": 0, "r": 100, "b": 50}},
                            {"text": "X", "bbox": {"l": 100, "t": 0, "r": 200, "b": 50}},
                        ]
                    ]
                }
            }
        ]
    }

    ocr_cells_debug = {
        "parsed_textline_cells": [
            {"text": "NEW | 123", "bbox": {"l": 10, "t": 10, "r": 90, "b": 40}},
        ]
    }

    changed = rewrite_docling_table_grid_text_from_ocr_cells(
        export_dict,
        ocr_cells_debug=ocr_cells_debug,
    )

    assert changed is True
    cell0 = export_dict["tables"][0]["data"]["grid"][0][0]
    assert cell0["text"] == "NEW 123"
