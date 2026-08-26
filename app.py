
import io
import base64
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
from PIL import Image

app = FastAPI(title="Industrial Defect Detection")

MODEL_PATH = "best.pt"  # place your trained model in the same folder
model = YOLO(MODEL_PATH)

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Industrial Defect Detection</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', -apple-system, sans-serif;
    background: #f6f8fa;
    min-height: 100vh;
    color: #1f2328;
    padding: 50px 20px;
  }}
  .container {{ max-width: 980px; margin: 0 auto; }}
  .badge {{
    display: inline-block;
    background: #e7f0fd;
    color: #1a56b8;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 16px;
    text-transform: uppercase;
  }}
  h1 {{
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
    color: #14212e;
  }}
  p.subtitle {{ color: #57606a; font-size: 14px; margin-bottom: 26px; }}
  .upload-box {{
    background: #ffffff;
    border: 1.5px dashed #c8d1da;
    border-radius: 14px;
    padding: 18px 24px;
    max-width: 460px;
    margin: 0 auto;
    text-align: center;
    box-shadow: 0 1px 2px rgba(20,33,46,0.04);
    transition: border-color 0.2s;
  }}
  .upload-box:hover {{ border-color: #2f6fd6; }}
  .hint {{ color: #8b949e; font-size: 11px; margin-top: 8px; }}
  input[type=file] {{
    color: #57606a;
    font-size: 13px;
    margin-bottom: 12px;
  }}
  input[type=file]::file-selector-button {{
    background: #f0f3f6;
    color: #1f2328;
    border: 1px solid #d0d7de;
    padding: 7px 14px;
    border-radius: 7px;
    font-size: 13px;
    cursor: pointer;
    margin-right: 12px;
  }}
  button {{
    display: block;
    margin: 12px auto 0;
    background: #2f6fd6;
    color: white;
    border: none;
    padding: 10px 26px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(47,111,214,0.35);
  }}
  button:hover {{ background: #2559ad; }}
  .result-section {{ margin-top: 34px; }}
  .result-title {{ font-size: 15px; color: #24292f; margin-bottom: 16px; font-weight: 700; }}
  .gallery {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px;
  }}
  .result-card {{
    background: #ffffff;
    border: 1px solid #d8dee4;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(20,33,46,0.05);
  }}
  .result-img {{
    width: 100%;
    max-height: 260px;
    object-fit: contain;
    background: #f0f3f6;
    display: block;
    border-bottom: 1px solid #d8dee4;
  }}
  .filename {{
    padding: 8px 12px 0;
    font-size: 12px;
    color: #8b949e;
  }}
  .detections {{ padding: 10px 12px 12px; display: flex; flex-direction: column; gap: 6px; }}
  .det-row {{
    background: #f6f8fa;
    border: 1px solid #e2e8ee;
    border-radius: 6px;
    padding: 7px 10px;
    display: flex;
    justify-content: space-between;
    font-size: 12px;
  }}
  .det-label {{ font-weight: 700; color: #1a56b8; }}
  .det-conf {{ color: #57606a; }}
  .no-defects {{ color: #1a7f37; font-weight: 700; font-size: 12px; padding: 10px 12px 12px; }}
  .empty-state {{ color: #8b949e; font-size: 13px; margin-top: 14px; text-align: center; }}
</style>
</head>
<body>
  <div class="container">
    <span class="badge">YOLOv8 &middot; Computer Vision</span>
    <h1>Industrial Defect Detection</h1>
    <p class="subtitle">Upload one or more steel/metal surface images to detect and classify surface defects.</p>

    <form action="/detect" method="post" enctype="multipart/form-data">
      <div class="upload-box">
        <input type="file" name="files" accept="image/*" multiple required><br>
        <button type="submit">Run Detection</button>
        <div class="hint">You can select multiple images at once</div>
      </div>
    </form>

    {result_html}
  </div>
</body>
</html>
"""


def build_card(filename: str, annotated_b64: str, detections: list) -> str:
    if not detections:
        det_html = '<div class="no-defects">No defects detected</div>'
    else:
        rows = "".join(
            f'<div class="det-row"><span class="det-label">{d["label"]}</span>'
            f'<span class="det-conf">{d["confidence"]:.2f}</span></div>'
            for d in detections
        )
        det_html = f'<div class="detections">{rows}</div>'

    return f"""
    <div class="result-card">
      <img class="result-img" src="data:image/jpeg;base64,{annotated_b64}">
      <div class="filename">{filename}</div>
      {det_html}
    </div>
    """


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE_TEMPLATE.format(result_html="")


@app.post("/detect", response_class=HTMLResponse)
async def detect(files: List[UploadFile] = File(...)):
    cards = []

    for file in files:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results = model(image)
        annotated = results[0].plot()  # numpy array, BGR

        annotated_rgb = Image.fromarray(annotated[:, :, ::-1])
        buf = io.BytesIO()
        annotated_rgb.save(buf, format="JPEG", quality=90)
        annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            detections.append({"label": label, "confidence": conf})

        cards.append(build_card(file.filename, annotated_b64, detections))

    result_html = f"""
    <div class="result-section">
      <div class="result-title">Detection Results ({len(cards)} image{'s' if len(cards) != 1 else ''})</div>
      <div class="gallery">{''.join(cards)}</div>
    </div>
    """
    return PAGE_TEMPLATE.format(result_html=result_html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)