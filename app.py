"""Table detection demo: upload a document page, see its tables boxed and cut out.

    pip install -r requirements.txt
    python app.py

Then open the address it prints (http://127.0.0.1:7860) in a browser.
"""
import gradio as gr
from ultralytics import YOLO

# The optimised model: OpenVINO INT8 at 800 px (see notebooks/05_optimization.ipynb).
MODEL = "results/05_optimization/openvino_int8_800_px_openvino_model"
IMGSZ = 800  # The model was exported for this input size.

model = YOLO(MODEL, task="detect")  # Loaded once, when the app starts.


def detect_tables(page, confidence):
    result = model.predict(page, imgsz=IMGSZ, conf=confidence, device="cpu", verbose=False)[0]

    boxed_page = result.plot()[:, :, ::-1]  # plot() returns BGR colours; the browser expects RGB.

    # Cut each table out of the original page, top to bottom.
    boxes = sorted(result.boxes.xyxy.tolist(), key=lambda box: box[1])
    tables = [page.crop(box) for box in boxes]  # box = [left, top, right, bottom] in pixels.

    milliseconds = sum(result.speed.values())  # Resize + model + removing overlapping boxes.
    summary = f"{len(tables)} table(s) found in {milliseconds:.0f} ms"
    return boxed_page, tables, summary


demo = gr.Interface(
    fn=detect_tables,
    inputs=[
        gr.Image(type="pil", label="Document page"),
        gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold"),
    ],
    outputs=[
        gr.Image(label="Detected tables"),
        gr.Gallery(label="Cut-out tables", format="png"),
        gr.Textbox(label="Result"),
    ],
    title="Table detection in document pages",
    description="Upload a page image (a scan or a screenshot of a PDF page). "
                "A lower threshold finds more tables but also more false boxes.",
    examples=[
        ["examples/financial_reports.png", 0.25],
        ["examples/scientific_articles.png", 0.25],
        ["examples/manuals.png", 0.25],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
