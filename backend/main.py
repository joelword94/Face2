import base64
import io
import json
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#: Rutas relativas a este archivo, no al cwd: asi el backend arranca igual
#: desde backend/ o desde la raiz del repo.
MODELOS_DIR = Path(__file__).resolve().parent.parent / "modelos"

YOLO_MODELS = {
    "yolov8n-oiv7": str(MODELOS_DIR / "yolov8n-oiv7.pt"),
    "yolov8s-oiv7": str(MODELOS_DIR / "yolov8s-oiv7.pt"),
}

GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

ALL_MODELS = list(YOLO_MODELS.keys()) + ["grounding-dino-tiny"]

_loaded_yolo = {}
_grounding_dino = {}


def get_yolo(name: str) -> YOLO:
    if name not in _loaded_yolo:
        _loaded_yolo[name] = YOLO(YOLO_MODELS[name])
    return _loaded_yolo[name]


def get_grounding_dino():
    if "processor" not in _grounding_dino:
        _grounding_dino["processor"] = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
        _grounding_dino["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_ID)
    return _grounding_dino["processor"], _grounding_dino["model"]


@app.get("/models")
def list_models():
    return {"models": ALL_MODELS}


def predict_yolo(image: Image.Image, model: str, conf: float):
    yolo = get_yolo(model)
    results = yolo.predict(image, conf=conf, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        detections.append(
            {
                "label": yolo.names[cls_id],
                "confidence": round(float(box.conf[0]), 4),
                "box": [round(v, 1) for v in box.xyxy[0].tolist()],
            }
        )

    annotated = result.plot()[:, :, ::-1]
    return detections, Image.fromarray(annotated)


def predict_grounding_dino(image: Image.Image, conf: float, prompt: str):
    processor, model = get_grounding_dino()
    text = prompt.strip()
    if not text.endswith("."):
        text += "."

    inputs = processor(images=image, text=text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=conf,
        text_threshold=conf,
        target_sizes=[image.size[::-1]],
    )[0]

    detections = []
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
        box_list = [round(v, 1) for v in box.tolist()]
        detections.append(
            {
                "label": label,
                "confidence": round(float(score), 4),
                "box": box_list,
            }
        )
        draw.rectangle(box.tolist(), outline="red", width=6)

    return detections, annotated


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("yolov8s-oiv7"),
    conf: float = Form(0.15),
    prompt: str = Form("shipping container"),
):
    if model not in ALL_MODELS:
        return {"error": f"modelo desconocido: {model}"}

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if model == "grounding-dino-tiny":
        detections, annotated_image = predict_grounding_dino(image, conf, prompt)
    else:
        detections, annotated_image = predict_yolo(image, model, conf)

    buf = io.BytesIO()
    annotated_image.save(buf, format="JPEG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "model": model,
        "detections": detections,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
    }


@app.post("/document/ocr")
async def document_ocr(file: UploadFile = File(...)):
    import document

    file_bytes = await file.read()
    pages = document.load_document_pages(file_bytes, file.filename)
    full_text, lines, paragraphs = document.run_ocr(pages)
    return {"full_text": full_text, "lines": lines, "paragraphs": paragraphs, "num_pages": len(pages)}


@app.post("/document/search-exact")
async def document_search_exact(
    lines: str = Form(...),
    query: str = Form(...),
):
    import document

    matches = document.exact_search(json.loads(lines), query)
    return {"query": query, "matches": matches}


@app.post("/document/search-semantic")
async def document_search_semantic(
    paragraphs: str = Form(...),
    query: str = Form(...),
):
    import document

    matches = document.semantic_search(json.loads(paragraphs), query)
    return {"query": query, "matches": matches}


@app.post("/document/seal")
async def document_seal(
    file: UploadFile = File(...),
    prompt: str = Form("stamp. seal."),
    conf: float = Form(0.2),
    reference_file: UploadFile | None = File(None),
):
    import document

    file_bytes = await file.read()
    pages = document.load_document_pages(file_bytes, file.filename)

    reference_image = None
    if reference_file is not None:
        ref_bytes = await reference_file.read()
        reference_image = document.load_document_pages(ref_bytes, reference_file.filename)[0]

    pages_result = document.detect_and_compare_seal(pages, prompt, conf, reference_image)

    response_pages = []
    all_detections = []
    for p in pages_result:
        buf = io.BytesIO()
        p["annotated_image"].save(buf, format="JPEG")
        annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        response_pages.append(
            {
                "page": p["page"],
                "detections": p["detections"],
                "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            }
        )
        all_detections.extend(p["detections"])

    return {
        "num_pages": len(pages),
        "pages": response_pages,
        "detections": all_detections,
    }


def _as_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/cargo/extract")
async def cargo_extract(
    file: UploadFile = File(...),
    use_semantic: bool = Form(True),
):
    """FOTO 1: lee la orden y extrae los campos estructurados."""
    import cargo
    import document

    file_bytes = await file.read()
    pages = document.load_document_pages(file_bytes, file.filename)
    full_text, lines, _ = document.run_ocr(pages)
    result = cargo.extract_order_fields(lines, use_semantic=use_semantic)

    return {
        "num_pages": len(pages),
        "full_text": full_text,
        "rows": result["rows"],
        "fields": result["fields"],
        "suggested_prompt": result["suggested_prompt"],
        "suggested_term_source": result["suggested_term_source"],
    }


@app.post("/cargo/verify")
async def cargo_verify(
    file: UploadFile = File(...),
    producto: str = Form(""),
    embalaje: str = Form(""),
    cantidad: str = Form(""),
    prompt_override: str = Form(""),
    reference_file: UploadFile | None = File(None),
    check_label: bool = Form(True),
    box_threshold: float = Form(0.30),
    text_threshold: float = Form(0.25),
    iou: float = Form(0.55),
    contain: float = Form(0.80),
    max_area_frac: float = Form(0.85),
    tolerance: float = Form(0.15),
):
    """FOTO 2: cuenta unidades de carga y verifica contra lo declarado."""
    import cargo
    import document

    file_bytes = await file.read()
    pages = document.load_document_pages(file_bytes, file.filename)
    image = cargo.preprocess_photo(pages[0])

    if prompt_override.strip():
        prompt, prompt_source = prompt_override.strip(), "override"
    else:
        prompt, prompt_source = cargo.map_packaging_term(embalaje)

    # --- Capa 1: conteo de la unidad de carga
    layer1 = cargo.count_load_units(
        image,
        prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        iou_thresh=iou,
        contain_thresh=contain,
        max_area_frac=max_area_frac,
    )
    if prompt_source == "fallback":
        layer1["reliability"] = "baja"
        layer1["reliability_reason"] = "el embalaje no se reconoció; se usó un término genérico"
    layer1["prompt_used"] = prompt
    layer1["prompt_source"] = prompt_source

    # --- Capa 2: identidad por etiqueta impresa
    if check_label and producto.strip():
        layer2 = cargo.verify_label_on_cargo(image, producto)
    else:
        layer2 = {"status": "no_aplica", "matched_text": None, "confidence": 0.0, "box": None, "all_text": ""}

    # --- Capa 3: comparación contra referencia
    reference_image = None
    if reference_file is not None:
        ref_bytes = await reference_file.read()
        if ref_bytes:
            reference_image = document.load_document_pages(ref_bytes, reference_file.filename)[0]
    layer3 = cargo.compare_against_reference(image, [d["box"] for d in layer1["detections"]], reference_image)

    expected, expected_unit, expected_countable = cargo.parse_quantity(cantidad)
    layer1["expected"] = expected

    verdict = cargo.compute_verdict(
        layer1, layer2, layer3, expected,
        expected_countable=expected_countable,
        tolerance=tolerance,
    )

    annotated = cargo.annotate_detections(image, layer1["detections"], layer1["suppressed"])

    return {
        "capa1": layer1,
        "capa2": layer2,
        "capa3": layer3,
        "expected": expected,
        "expected_unit": expected_unit,
        "expected_countable": expected_countable,
        "tolerance": tolerance,
        "image_size": list(image.size),
        "annotated_image": _as_data_url(annotated),
        **verdict,
    }
