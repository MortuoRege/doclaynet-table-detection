# Validation pages

The 500 validation pages used to score every model, so the failure analysis and
optimisation notebooks run without downloading anything.

- `validation_pages/images/<page_hash>.png`: the page images, unchanged.
- `validation_pages/labels/<page_hash>.txt`: table boxes in YOLO format
  (`0 center_x center_y width height`, each divided by the page size).
  An empty file means the page has no table.

Which pages these are, and how they were chosen, is in
`results/01_scaling_study/selected_pages.json` and `notebooks/01_scaling_study.ipynb`.

**Source:** [DocLayNet v1.2](https://huggingface.co/datasets/docling-project/DocLayNet-v1.2)
(revision `85cadd47c3ac59e1a2c3677ec72a02b143d96430`), by IBM Research,
licensed under CDLA-Permissive-1.0. Only the table annotations are kept.
