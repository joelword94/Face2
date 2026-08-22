import io
import unicodedata

import numpy as np
import torch
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

_easyocr_reader = None
_embedder = None
_clip = {}


def get_ocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr

        _easyocr_reader = easyocr.Reader(["es", "en"], gpu=False)
    return _easyocr_reader


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embedder


def get_clip():
    if "model" not in _clip:
        _clip["model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip["processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip["model"], _clip["processor"]


def load_document_pages(file_bytes: bytes, filename: str) -> list[Image.Image]:
    if filename.lower().endswith(".pdf"):
        import pymupdf

        pdf = pymupdf.open(stream=file_bytes, filetype="pdf")
        pages = []
        for page in pdf:
            pix = page.get_pixmap(dpi=200)
            pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        return pages
    return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]


def run_ocr_single_page(image: Image.Image):
    reader = get_ocr_reader()
    raw = reader.readtext(np.array(image))

    lines = []
    for box, text, conf in raw:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        lines.append(
            {
                "text": str(text),
                "confidence": round(float(conf), 4),
                "box": [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)],
            }
        )
    lines.sort(key=lambda l: (l["box"][1], l["box"][0]))

    paragraphs = []
    current = None
    for line in lines:
        y_top = line["box"][1]
        height = line["box"][3] - line["box"][1]
        if current is None:
            current = {"text": line["text"], "box": list(line["box"]), "lines": [line]}
        else:
            prev_bottom = current["box"][3]
            gap = y_top - prev_bottom
            if gap < height * 1.2:
                current["text"] += " " + line["text"]
                current["box"][0] = min(current["box"][0], line["box"][0])
                current["box"][1] = min(current["box"][1], line["box"][1])
                current["box"][2] = max(current["box"][2], line["box"][2])
                current["box"][3] = max(current["box"][3], line["box"][3])
                current["lines"].append(line)
            else:
                paragraphs.append(current)
                current = {"text": line["text"], "box": list(line["box"]), "lines": [line]}
    if current:
        paragraphs.append(current)

    for p in paragraphs:
        p["box"] = [round(v, 1) for v in p["box"]]
        del p["lines"]

    full_text = "\n".join(l["text"] for l in lines)
    return full_text, lines, paragraphs


def run_ocr(pages: list[Image.Image]):
    all_lines = []
    all_paragraphs = []
    full_text_parts = []

    for page_num, image in enumerate(pages, start=1):
        full_text, lines, paragraphs = run_ocr_single_page(image)

        for line in lines:
            line["page"] = page_num
        for p in paragraphs:
            p["page"] = page_num

        all_lines.extend(lines)
        all_paragraphs.extend(paragraphs)
        full_text_parts.append(f"--- Página {page_num} ---\n{full_text}")

    return "\n\n".join(full_text_parts), all_lines, all_paragraphs


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def exact_search(lines, query: str):
    norm_query = _normalize(query)
    if not norm_query:
        return []
    matches = []
    for line in lines:
        if norm_query in _normalize(line["text"]):
            matches.append(line)
    return matches


def semantic_search(paragraphs, query: str, top_k: int = 5):
    if not paragraphs:
        return []
    model = get_embedder()
    texts = [p["text"] for p in paragraphs]
    para_embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    scores = para_embeddings @ query_embedding
    order = np.argsort(-scores)[:top_k]

    results = []
    for idx in order:
        results.append(
            {
                "text": paragraphs[idx]["text"],
                "box": paragraphs[idx]["box"],
                "similarity": round(float(scores[idx]), 4),
            }
        )
    return results


def _detect_and_compare_seal_single_page(image: Image.Image, prompt: str, threshold: float, reference_embedding, clip_model, clip_processor):
    from main import get_grounding_dino

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
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    detections = []
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
        box_list = [round(v, 1) for v in box.tolist()]
        entry = {
            "label": label,
            "confidence": round(float(score), 4),
            "box": box_list,
        }

        if reference_embedding is not None:
            crop = image.crop(box.tolist())
            crop_inputs = clip_processor(images=crop, return_tensors="pt")
            with torch.no_grad():
                crop_embedding = clip_model.get_image_features(**crop_inputs).pooler_output
            crop_embedding = crop_embedding / crop_embedding.norm(dim=-1, keepdim=True)
            similarity = float((crop_embedding @ reference_embedding.T).item())
            entry["match_similarity"] = round(similarity, 4)

        detections.append(entry)
        color = "lime" if reference_embedding is None else ("lime" if entry.get("match_similarity", 0) > 0.8 else "red")
        draw.rectangle(box.tolist(), outline=color, width=6)

    return detections, annotated


def detect_and_compare_seal(pages: list[Image.Image], prompt: str, threshold: float, reference_image: Image.Image | None):
    clip_model, clip_processor = get_clip() if reference_image is not None else (None, None)
    reference_embedding = None
    if reference_image is not None:
        ref_inputs = clip_processor(images=reference_image, return_tensors="pt")
        with torch.no_grad():
            reference_embedding = clip_model.get_image_features(**ref_inputs).pooler_output
        reference_embedding = reference_embedding / reference_embedding.norm(dim=-1, keepdim=True)

    pages_result = []
    for page_num, image in enumerate(pages, start=1):
        detections, annotated = _detect_and_compare_seal_single_page(
            image, prompt, threshold, reference_embedding, clip_model, clip_processor
        )
        for d in detections:
            d["page"] = page_num
        pages_result.append({"page": page_num, "detections": detections, "annotated_image": annotated})

    return pages_result
