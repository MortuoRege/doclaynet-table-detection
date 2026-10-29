# Table Detection in Document Pages

Finds every table on a document page image, draws a box around it and cuts it
out. A small YOLO11n model, fine-tuned on 18,000 pages of the DocLayNet dataset
and optimised to run on a CPU in about 70 ms per page.

Built for the Upgaming Talent Growth Program (Machine Learning Engineer,
Computer Vision). All final models were trained on free Kaggle GPUs.

> **Technical report:** [REPORT.md](REPORT.md)
>
> პროექტის დოკუმენტაცია ინგლისურად დავწერე, რადგან კოდი და ტექნიკური ტერმინები
> ინგლისურადაა. საჭიროების შემთხვევაში ქართულ ვერსიასაც მოგაწვდით.

![A manual page with its table detected](results/demo_detection.png)

## Results

Validation set: 500 pages from 127 documents never seen in training, with 180 tables.

| Model | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| 640 px, 20 epochs (baseline) | 0.845 | 0.753 | 0.890 | 0.767 |
| 896 px, 30 epochs | 0.852 | 0.756 | 0.857 | 0.802 |
| **1024 px, 60 epochs (final)** | **0.894** | **0.809** | **0.920** | **0.835** |

Final model on the **test set**, scored once (500 pages from 139 other unseen
documents, 200 tables): **mAP50 0.927, mAP50-95 0.869**, precision 0.950,
recall 0.860.

Optimised for CPU, without retraining (Intel i7-12700H, one page at a time):

| Variant | Size | Time per page | mAP50-95 |
|---|---|---|---|
| PyTorch FP32 (before) | 5.6 MB | 373 ms | 0.809 |
| **OpenVINO INT8, 800 px (used in the app)** | **3.4 MB** | **73 ms** | **0.804** |

5.1× faster for −0.005 mAP50-95. All variants: [REPORT.md](REPORT.md), section 6.

## What is in this repository

```
├── app.py                  Demo app: upload a page, see tables boxed and cut out
├── REPORT.md               Technical report
├── requirements.txt        Packages for running locally
├── examples/               Three sample pages for the app
├── models/
│   └── best_1024_60ep.pt   The final trained model (PyTorch)
├── notebooks/
│   ├── 01_scaling_study.ipynb             Data selection + training on 3k / 10k / 18k pages (Kaggle)
│   ├── 02_resolution_896px_30ep.ipynb     Higher resolution, more epochs (Kaggle)
│   ├── 03_final_model_1024px_60ep.ipynb   The final model (Kaggle)
│   ├── 04_failure_analysis.ipynb          Where and why the model fails (CPU)
│   ├── 05_optimization.ipynb              ONNX / OpenVINO / INT8 comparison (CPU)
│   └── extra/1024px_60ep_rented_gpu.ipynb The same final model, first trained on a rented GPU
├── results/                Scores, training curves and plots of every experiment
│   └── 05_optimization/    Also holds the exported models, incl. the one the app uses
└── data/validation_pages/  The 500 validation pages (images + labels)
```

## How to run

### Demo app (CPU, no GPU needed)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860, upload a page image or click an example. The
confidence slider trades recall (lower) against precision (higher).

### Use the model in your own code

```python
from ultralytics import YOLO

model = YOLO("results/05_optimization/openvino_int8_800_px_openvino_model", task="detect")
result = model.predict("page.png", imgsz=800, conf=0.25)[0]
for left, top, right, bottom in result.boxes.xyxy.tolist():
    print("table at", round(left), round(top), round(right), round(bottom))
```

Or the original PyTorch model: `YOLO("models/best_1024_60ep.pt")` with `imgsz=1024`.

### Retrain the model (Kaggle, free)

1. Create a Kaggle notebook and import `notebooks/03_final_model_1024px_60ep.ipynb`.
2. In the settings, turn on a **GPU** (T4 or P100) and **Internet**.
3. Click **Save Version → Save & Run All**.

The notebook downloads its own data (about 1 hour, no account needed), trains
for up to 60 epochs (about 7 hours on a T4) and saves the model to the version's
**Output** tab. The first two notebooks run the same way.

### Failure analysis and optimisation (CPU)

Open `notebooks/04_failure_analysis.ipynb` or `notebooks/05_optimization.ipynb`
in Jupyter, from inside the `notebooks/` folder, and run all cells. They use the
validation pages in `data/` and need no internet. Optimisation takes about
15–30 minutes.

## Data

[DocLayNet v1.2](https://huggingface.co/datasets/docling-project/DocLayNet-v1.2)
(IBM Research, CDLA-Permissive-1.0), pinned to one revision. From about 80,000
annotated pages, the notebooks select 18,000 training, 500 validation and 500
test pages: at most 10 pages per document, stratified by document type and by
whether the page has a table, with no document shared between splits. Only the
"Table" class is kept. Details: [REPORT.md](REPORT.md), section 1.
