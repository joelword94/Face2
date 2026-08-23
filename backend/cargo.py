"""Verificación de carga contra orden de compra ("Container 2").

Tres capas de verificación sobre la foto del cargamento:
  Capa 1 - conteo de la UNIDAD DE CARGA (embalaje), no del producto.
  Capa 2 - identidad del producto leyendo la etiqueta impresa (OCR).
  Capa 3 - comparación visual contra una foto de referencia (CLIP).

El objetivo de detección es siempre el embalaje. Un producto como "pitahaya"
no es detectable en la foto porque viene dentro de cajas; lo contable son
las cajas.
"""

import difflib
import math
import re

import numpy as np
import torch
from PIL import Image, ImageDraw

import document

# ---------------------------------------------------------------------------
# Capa 1 - preproceso y deduplicación
# ---------------------------------------------------------------------------


def preprocess_photo(image: Image.Image, max_side: int = 1333) -> Image.Image:
    """Reduce la foto antes de inferir.

    Las fotos de celular vienen en 4000-5000 px; Grounding DINO reescala
    internamente a 800 de todos modos, y devolver la anotada en tamaño
    original infla la respuesta base64 a varios MB.
    """
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def _areas(boxes: np.ndarray) -> np.ndarray:
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def _pair_intersection(boxes: np.ndarray, i: int) -> np.ndarray:
    x1 = np.maximum(boxes[i, 0], boxes[:, 0])
    y1 = np.maximum(boxes[i, 1], boxes[:, 1])
    x2 = np.minimum(boxes[i, 2], boxes[:, 2])
    y2 = np.minimum(boxes[i, 3], boxes[:, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """NMS greedy en numpy puro."""
    order = np.argsort(-scores)
    areas = _areas(boxes)
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        inter = _pair_intersection(boxes, i)[rest]
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thresh]
    return keep


def dedupe_detections(
    detections: list[dict],
    image_size: tuple[int, int],
    iou_thresh: float = 0.55,
    contain_thresh: float = 0.80,
    max_area_frac: float = 0.85,
    min_area_frac: float = 0.0015,
    area_ratio_min: float = 1.8,
) -> tuple[list[dict], list[dict]]:
    """Filtra detecciones solapadas en tres etapas.

    NMS por sí solo NO sirve: la caja que cubre toda la imagen tiene IoU ~0.18
    contra un objeto real, así que sobrevive y infla el conteo. La etapa de
    contención es la que la elimina.

    Devuelve (conservadas, suprimidas); cada suprimida lleva el motivo.
    """
    if not detections:
        return [], []

    boxes = np.array([d["box"] for d in detections], dtype=float)
    scores = np.array([d["confidence"] for d in detections], dtype=float)
    areas = _areas(boxes)
    page_area = float(image_size[0] * image_size[1]) or 1.0
    frac = areas / page_area

    suppressed: list[dict] = []
    alive = np.ones(len(detections), dtype=bool)

    def kill(idx: int, reason: str) -> None:
        alive[idx] = False
        entry = dict(detections[idx])
        entry["reason"] = reason
        suppressed.append(entry)

    # --- Etapa A: área absoluta -------------------------------------------
    too_big = np.where(frac > max_area_frac)[0]
    if len(too_big) == len(detections):
        # Todo se rechazaría: probablemente un objeto en primer plano que
        # llena el encuadre. Nunca dejar que un filtro lleve el conteo a 0.
        best = int(np.argmax(scores))
        for i in too_big:
            if int(i) != best:
                kill(int(i), "area_too_large")
    else:
        for i in too_big:
            kill(int(i), "area_too_large")

    for i in np.where(frac < min_area_frac)[0]:
        if alive[i]:
            kill(int(i), "area_too_small")

    # --- Etapa B: contención (padre con >=2 hijos) -------------------------
    live_idx = [i for i in range(len(detections)) if alive[i]]
    to_drop = []
    for i in live_idx:
        inter = _pair_intersection(boxes, i)
        children = 0
        for j in live_idx:
            if i == j:
                continue
            smaller = min(areas[i], areas[j])
            if smaller <= 0:
                continue
            if inter[j] / smaller > contain_thresh and areas[i] > areas[j] * area_ratio_min:
                children += 1
        if children >= 2:
            to_drop.append(i)
    for i in to_drop:
        kill(i, "contains_children")

    # --- Etapa C: NMS por IoU ---------------------------------------------
    live_idx = [i for i in range(len(detections)) if alive[i]]
    if live_idx:
        sub_boxes = boxes[live_idx]
        sub_scores = scores[live_idx]
        keep_local = set(_nms(sub_boxes, sub_scores, iou_thresh))
        for pos, i in enumerate(live_idx):
            if pos not in keep_local:
                kill(i, "iou_duplicate")

    kept = [dict(detections[i]) for i in range(len(detections)) if alive[i]]

    # Red de seguridad: si todo se filtró, conservar la de mayor score.
    if not kept and detections:
        best = int(np.argmax(scores))
        kept = [dict(detections[best])]
        suppressed = [s for s in suppressed if s.get("box") != detections[best]["box"]]

    kept.sort(key=lambda d: (d["box"][1], d["box"][0]))
    for n, d in enumerate(kept, start=1):
        d["index"] = n
    return kept, suppressed


def _run_grounding_dino(image: Image.Image, prompt: str, box_threshold: float, text_threshold: float) -> list[dict]:
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
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    return [
        {
            "label": label,
            "confidence": round(float(score), 4),
            "box": [round(v, 1) for v in box.tolist()],
        }
        for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"])
    ]


def count_load_units(
    image: Image.Image,
    prompt: str,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
    **dedupe_kwargs,
) -> dict:
    """Capa 1: detecta y cuenta unidades de carga."""
    raw = _run_grounding_dino(image, prompt, box_threshold, text_threshold)
    kept, suppressed = dedupe_detections(raw, image.size, **dedupe_kwargs)

    reliability = "media"
    reason = None
    if not raw:
        reliability = "baja"
        reason = "el modelo no detectó nada con ese término"
    elif len(raw) > 1 and len(suppressed) >= len(raw) * 0.6:
        reliability = "baja"
        reason = f"{len(suppressed)} de {len(raw)} detecciones fueron descartadas por solapamiento"
    elif suppressed:
        reason = f"{len(suppressed)} de {len(raw)} detecciones fueron descartadas por solapamiento"

    return {
        "count": len(kept),
        "raw_count": len(raw),
        "detections": kept,
        "suppressed": suppressed,
        "reliability": reliability,
        "reliability_reason": reason,
    }


def annotate_detections(image: Image.Image, kept: list[dict], suppressed: list[dict]) -> Image.Image:
    """Dibuja las conservadas en verde numeradas y las suprimidas en gris fino."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for d in suppressed:
        draw.rectangle(d["box"], outline="#888888", width=1)
    for d in kept:
        draw.rectangle(d["box"], outline="lime", width=4)
        x, y = d["box"][0], d["box"][1]
        draw.text((x + 5, max(0, y + 3)), str(d.get("index", "")), fill="lime")
    draw.text((8, 8), f"conteo: {len(kept)}", fill="lime")
    return out


# ---------------------------------------------------------------------------
# Capa 2 - identidad del producto por etiqueta impresa
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "de", "del", "la", "el", "los", "las", "y", "con", "sin", "para", "por",
    "en", "a", "al", "un", "una", "tipo", "clase", "primera", "segunda",
    "calidad", "marca", "kg", "kgs", "gr", "unidad", "unidades",
}


def _significant_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", document._normalize(text))
    return [t for t in tokens if len(t) >= 4 and t not in _STOPWORDS]


#: Se busca sobre texto de confianza muy baja porque el matching difuso puede
#: recuperar "Fuit" -> "fruit" en fuentes cursivas. Pero para ACUSAR una
#: discrepancia se exige texto de alta confianza: un falso "no coincide"
#: detiene un camión legítimo, mientras que un "no legible" solo pide que un
#: humano revise. La asimetría de costos manda.
LABEL_SEARCH_MIN_CONF = 0.05
LABEL_TRUST_MIN_CONF = 0.50
LABEL_TRUST_MIN_CHARS = 25


def verify_label_on_cargo(image: Image.Image, product_name: str, min_ratio: float = 0.80) -> dict:
    """Capa 2: busca el nombre del producto en el texto impreso de la carga.

    Estados:
      encontrado        - se leyó un token del producto (aunque sea con OCR ruidoso)
      no_legible        - no se leyó texto confiable suficiente para concluir
      producto_distinto - se leyó bastante texto CONFIABLE y nada del producto
    """
    base = {"status": "no_legible", "matched_text": None, "confidence": 0.0, "box": None, "all_text": ""}
    if not product_name or not product_name.strip():
        base["status"] = "no_aplica"
        return base

    _, lines, _ = document.run_ocr_single_page(image)
    searchable = [l for l in lines if l["confidence"] >= LABEL_SEARCH_MIN_CONF]
    all_text = " ".join(l["text"] for l in searchable)
    base["all_text"] = all_text

    wanted = _significant_tokens(product_name)
    if not wanted:
        base["status"] = "no_aplica"
        return base

    best = None
    for line in searchable:
        for token in re.findall(r"[a-z0-9]+", document._normalize(line["text"])):
            if len(token) < 4:
                continue
            for w in wanted:
                ratio = difflib.SequenceMatcher(None, w, token).ratio()
                if ratio >= min_ratio and (best is None or ratio > best[0]):
                    best = (ratio, line)

    if best is not None:
        ratio, line = best
        return {
            "status": "encontrado",
            "matched_text": line["text"],
            # Ponderar por la confianza real del OCR: si la etiqueta se leyó
            # mal, el número reportado debe reflejarlo.
            "confidence": round(ratio * line["confidence"], 4),
            "box": line["box"],
            "all_text": all_text,
        }

    # No se encontró el producto. Solo se acusa discrepancia si de verdad se
    # leyó texto confiable; si el OCR solo produjo ruido, es "no legible".
    trusted = "".join(
        document._normalize(l["text"]).replace(" ", "")
        for l in lines
        if l["confidence"] >= LABEL_TRUST_MIN_CONF
    )
    base["status"] = "producto_distinto" if len(trusted) >= LABEL_TRUST_MIN_CHARS else "no_legible"
    return base


# ---------------------------------------------------------------------------
# Capa 3 - comparación visual contra referencia
# ---------------------------------------------------------------------------


#: Calibrado midiendo casos reales en pruebas/:
#:   referencia = sello de documento (nada que ver) -> 0.42
#:   referencia = contenedor (otro embalaje)        -> 0.56
#:   referencia = la caja correcta                  -> 0.78
#: Ojo: son 3 muestras, y la escala absoluta de similitud de CLIP depende
#: mucho del dominio. Re-calibrar con fotos reales de garita antes de confiar
#: en estos cortes.
REF_COINCIDE_MIN = 0.72
REF_REVISION_MIN = 0.62


def compare_against_reference(image: Image.Image, boxes: list[list[float]], reference_image: Image.Image) -> dict:
    """Capa 3: similitud CLIP entre cada recorte y la foto de referencia."""
    if reference_image is None or not boxes:
        return {"status": "no_aplica", "similarity": None, "per_box": []}

    clip_model, clip_processor = document.get_clip()

    ref_inputs = clip_processor(images=reference_image, return_tensors="pt")
    with torch.no_grad():
        ref_emb = clip_model.get_image_features(**ref_inputs).pooler_output
    ref_emb = ref_emb / ref_emb.norm(dim=-1, keepdim=True)

    sims = []
    for box in boxes:
        crop = image.crop(tuple(box))
        if crop.width < 4 or crop.height < 4:
            continue
        crop_inputs = clip_processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            emb = clip_model.get_image_features(**crop_inputs).pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sims.append(round(float((emb @ ref_emb.T).item()), 4))

    if not sims:
        return {"status": "no_aplica", "similarity": None, "per_box": []}

    best = max(sims)
    if best > REF_COINCIDE_MIN:
        status = "coincide"
    elif best > REF_REVISION_MIN:
        status = "revision"
    else:
        status = "no_coincide"
    return {"status": status, "similarity": best, "per_box": sims}


# ---------------------------------------------------------------------------
# Extracción de campos del documento
# ---------------------------------------------------------------------------


def group_lines_into_rows(lines: list[dict]) -> list[dict]:
    """Reagrupa las líneas de OCR en filas visuales.

    Necesario porque document.run_ocr ordena por y_top crudo, y un jitter de
    pocos píxeles invierte el orden de lectura dentro de una misma fila. Aquí
    se agrupa por solapamiento vertical y se ordena por X.
    """
    rows: list[dict] = []
    by_page: dict[int, list[dict]] = {}
    for idx, line in enumerate(lines):
        by_page.setdefault(line.get("page", 1), []).append((idx, line))

    for page in sorted(by_page):
        items = sorted(by_page[page], key=lambda p: (p[1]["box"][1] + p[1]["box"][3]) / 2)
        current: list[tuple[int, dict]] = []
        cur_top = cur_bottom = None

        def flush() -> None:
            if not current:
                return
            ordered = sorted(current, key=lambda p: p[1]["box"][0])
            boxes = [p[1]["box"] for p in ordered]
            confs = [p[1]["confidence"] for p in ordered]
            rows.append(
                {
                    "text": " ".join(p[1]["text"] for p in ordered),
                    "box": [
                        round(min(b[0] for b in boxes), 1),
                        round(min(b[1] for b in boxes), 1),
                        round(max(b[2] for b in boxes), 1),
                        round(max(b[3] for b in boxes), 1),
                    ],
                    "page": page,
                    "ocr_confidence": round(sum(confs) / len(confs), 4),
                    "line_indices": [p[0] for p in ordered],
                }
            )

        for idx, line in items:
            top, bottom = line["box"][1], line["box"][3]
            if not current:
                current = [(idx, line)]
                cur_top, cur_bottom = top, bottom
                continue
            overlap = min(bottom, cur_bottom) - max(top, cur_top)
            min_height = min(bottom - top, cur_bottom - cur_top)
            if min_height > 0 and overlap > 0.5 * min_height:
                current.append((idx, line))
                cur_top, cur_bottom = min(cur_top, top), max(cur_bottom, bottom)
            else:
                flush()
                current = [(idx, line)]
                cur_top, cur_bottom = top, bottom
        flush()

    return rows


COUNTABLE_UNITS = {
    "caja", "cajas", "saco", "sacos", "costal", "costales", "bulto", "bultos",
    "gaveta", "gavetas", "jaba", "jabas", "java", "javas", "canasta", "canastas",
    "pallet", "pallets", "palet", "palets", "paleta", "paletas", "tarima", "tarimas",
    "pieza", "piezas", "pza", "pzas", "unidad", "unidades", "und", "uds", "u",
    "tambor", "tambores", "barril", "barriles", "bidon", "bidones",
    "huacal", "huacales", "guacal", "guacales", "jaula", "jaulas",
    "rollo", "rollos", "bobina", "bobinas", "cilindro", "cilindros",
    "tubo", "tubos", "atado", "atados", "paquete", "paquetes", "funda", "fundas",
    "quintal", "quintales", "contenedor", "contenedores",
}

NON_COUNTABLE_UNITS = {
    "kg", "kgs", "kilo", "kilos", "kilogramo", "kilogramos",
    "ton", "tn", "tonelada", "toneladas", "t",
    "lb", "lbs", "libra", "libras", "g", "gr", "gramo", "gramos",
    "m3", "m2", "metro", "metros", "ml", "cc",
    "lt", "l", "litro", "litros", "galon", "galones", "gal",
    "usd", "dolares",
}

# Unidades que NO son carga: tiempo, plazos, condiciones contractuales.
# Sin esto, "vigencia de 12 meses" se interpreta como 12 bultos.
EXCLUDED_UNITS = {
    "mes", "meses", "dia", "dias", "ano", "anos", "anio", "anios", "year",
    "semana", "semanas", "hora", "horas", "minuto", "minutos",
    "vez", "veces", "entrega", "entregas", "cuota", "cuotas", "parcial", "parciales",
    "por ciento", "porciento", "pct",
}

_UNIT_ALTERNATION = "|".join(
    sorted(COUNTABLE_UNITS | NON_COUNTABLE_UNITS | EXCLUDED_UNITS, key=len, reverse=True)
)

RE_QTY = re.compile(
    r"(?<![\w./-])(?P<num>\d{1,3}(?:[.\s]\d{3})+|\d+(?:[,.]\d{1,2})?)\s*(?P<unit>" + _UNIT_ALTERNATION + r")?(?![\w])",
    re.IGNORECASE,
)

RE_PLATE = re.compile(r"\b(?P<plate>[A-Z]{2,3})[\s\-.]?(?P<digits>\d{3,4})\b", re.IGNORECASE)

RE_ORDER = re.compile(
    r"(?:orden|oc|o/c|pedido|guia|folio|factura|autorizacion|no|nro|num|n|#)\s*[:.\-#]?\s*"
    r"(?P<id>[A-Z0-9][A-Z0-9\-/]{2,19})",
    re.IGNORECASE,
)

PLATE_BLOCKLIST = {"ruc", "iva", "usd", "sri", "nit", "ci", "cid", "tel", "fax"}

# Filas monetarias: nunca son la cantidad de carga. Sin esto, "Valor total:
# USD 3.750,00" se interpreta como 3750 bultos.
MONEY_HINTS = {
    "usd", "us", "dolar", "dolares", "valor", "precio", "subtotal", "iva",
    "costo", "monto", "pago", "pagar", "importe", "tarifa", "flete", "total a",
}

_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8", "Z": "2", "G": "6"})


def _fix_digits(token: str) -> str:
    digits = sum(c.isdigit() for c in token)
    if token and digits / len(token) >= 0.6:
        return token.translate(_DIGIT_FIX)
    return token


def parse_quantity(text: str) -> tuple[int | None, str | None, bool]:
    """Extrae (numero, unidad, es_contable) del texto.

    es_contable=False para pesos y volúmenes: "500 kg" no se puede contar en
    una foto. Ese caso obliga a pedir la cantidad de bultos al usuario.
    """
    if not text:
        return None, None, False

    fixed = " ".join(_fix_digits(t) for t in str(text).split())
    best: tuple[int, str | None, bool] | None = None

    for m in RE_QTY.finditer(fixed):
        raw = m.group("num")
        unit = (m.group("unit") or "").lower() or None

        # "12 meses", "24 entregas": no son carga.
        if unit in EXCLUDED_UNITS:
            continue

        # Separador de miles vs decimal: ".500" con exactamente 3 digitos es
        # separador (Ecuador escribe 1.500); ".5" es decimal.
        cleaned = raw.replace(" ", "")
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", cleaned):
            value_str = re.sub(r"[.,]", "", cleaned)
        else:
            value_str = re.split(r"[.,]", cleaned)[0]
        try:
            value = int(value_str)
        except ValueError:
            continue

        countable = unit in COUNTABLE_UNITS if unit else True
        candidate = (value, unit, countable)
        # Preferir una cantidad contable con unidad explícita.
        if best is None:
            best = candidate
        elif countable and unit and not (best[2] and best[1]):
            best = candidate

    if best is None:
        return None, None, False
    return best


PACKAGING_ES_EN = {
    "caja de carton": "cardboard box", "cajas de carton": "cardboard box",
    "caja": "cardboard box", "cajas": "cardboard box",
    "carton": "cardboard box", "cartones": "cardboard box",
    "saco": "sack", "sacos": "sack", "costal": "sack", "costales": "sack",
    "bulto": "sack", "bultos": "sack", "quintal": "sack", "quintales": "sack",
    "gaveta": "plastic crate", "gavetas": "plastic crate",
    "jaba": "plastic crate", "jabas": "plastic crate",
    "java": "plastic crate", "javas": "plastic crate",
    "canasta": "plastic crate", "canastas": "plastic crate",
    "pallet": "pallet", "pallets": "pallet", "palet": "pallet", "palets": "pallet",
    "paleta": "pallet", "paletas": "pallet", "tarima": "pallet", "tarimas": "pallet",
    "tambor": "metal drum", "tambores": "metal drum",
    "barril": "metal drum", "barriles": "metal drum",
    "bidon": "plastic jerrycan", "bidones": "plastic jerrycan",
    "garrafa": "plastic jerrycan", "garrafas": "plastic jerrycan",
    "big bag": "bulk bag", "maxisaco": "bulk bag", "supersaco": "bulk bag",
    "huacal": "wooden crate", "guacal": "wooden crate",
    "jaula": "wooden crate", "jaulas": "wooden crate",
    "rollo": "roll", "rollos": "roll",
    "bobina": "steel coil", "bobinas": "steel coil",
    "cilindro": "gas cylinder", "cilindros": "gas cylinder",
    "tubo": "steel pipe", "tubos": "steel pipe",
    "contenedor": "shipping container", "contenedores": "shipping container",
}

PACKAGING_FALLBACK = "cardboard box"


def map_packaging_term(spanish_text: str) -> tuple[str, str]:
    """Traduce el embalaje a un término en inglés para Grounding DINO.

    El modelo está entrenado en inglés; pasarle español produce cajas casi
    aleatorias, lo cual es una falla silenciosa. Si no se reconoce el término
    se devuelve el embalaje más común y se marca como fallback para que la UI
    avise y pida override.
    """
    if not spanish_text or not spanish_text.strip():
        return PACKAGING_FALLBACK, "fallback"

    norm = document._normalize(spanish_text)
    if norm in PACKAGING_ES_EN:
        return PACKAGING_ES_EN[norm], "mapped"

    for key in sorted(PACKAGING_ES_EN, key=len, reverse=True):
        if key in norm:
            return PACKAGING_ES_EN[key], "mapped_partial"

    for token in re.findall(r"[a-z]+", norm):
        if token in PACKAGING_ES_EN:
            return PACKAGING_ES_EN[token], "mapped_partial"

    return PACKAGING_FALLBACK, "fallback"


FIELD_LABELS = {
    "producto": [
        "descripcion del producto", "descripcion de la carga", "descripcion",
        "producto", "mercaderia", "mercancia", "material", "articulo",
        "item", "detalle", "carga", "bien", "insumo",
    ],
    "embalaje": [
        "tipo de bulto", "unidad de medida", "presentacion", "embalaje",
        "empaque", "envase", "contenido", "unidad",
    ],
    "cantidad": [
        "numero de unidades", "nro de unidades", "no de unidades",
        "cantidad total", "total bultos", "total piezas", "peso neto",
        "peso bruto", "cantidad", "cant", "unidades", "piezas", "pzas",
        "bultos", "sacos", "cajas", "total", "peso", "volumen",
    ],
    "vehiculo": [
        "unidad de transporte", "numero de placa", "nro de placa", "no de placa",
        "placa", "placas", "vehiculo", "camion", "transporte", "tracto",
        "cabezal", "remolque", "furgon", "matricula",
    ],
    "orden": [
        "numero de orden", "nro de orden", "no de orden", "orden de compra",
        "guia de remision", "autorizacion", "orden", "pedido", "guia",
        "documento", "folio", "factura", "oc", "o/c",
    ],
}

SEMANTIC_QUERIES = {
    "producto": "descripcion del producto o material transportado",
    "embalaje": "tipo de embalaje o presentacion de la carga",
    "cantidad": "cantidad total de unidades o piezas",
    "vehiculo": "placa del vehiculo o unidad de transporte",
    "orden": "numero de orden de compra o autorizacion",
}

METHOD_BASE = {"label_exact": 0.90, "label_fuzzy": 0.70, "semantic": 0.50, "regex_only": 0.35}

SEMANTIC_MIN_SIMILARITY = 0.25


def _has_money_hint(text: str) -> bool:
    norm = document._normalize(text)
    if "$" in text:
        return True
    tokens = set(re.findall(r"[a-z]+", norm))
    return any(h in tokens or h in norm for h in MONEY_HINTS)


def _empty_field() -> dict:
    return {
        "value": None, "number": None, "unit": None, "es_contable": False,
        "raw": None, "confidence": 0.0, "method": "none", "strategy": None,
        "source_text": None, "source_box": None, "page": None, "candidates": [],
    }


def _find_label_rows(rows: list[dict], field: str) -> list[tuple[dict, str, str]]:
    """Todas las filas que contienen una etiqueta del campo, en orden de prioridad.

    Devuelve una lista y no un único match porque un documento puede traer
    varias etiquetas del mismo campo ("Cantidad: 18 cajas" y "Peso neto: 90 kg");
    quien llama decide cuál prefiere.
    """
    found: list[tuple[dict, str, str]] = []
    seen: set[int] = set()

    for synonym in FIELD_LABELS[field]:
        norm_syn = document._normalize(synonym)
        for row in rows:
            if id(row) in seen:
                continue
            if norm_syn in document._normalize(row["text"]):
                found.append((row, "label_exact", synonym))
                seen.add(id(row))

    if found:
        return found

    # Fuzzy por token, tolera "Cantiad" / "C ntidad"
    for synonym in FIELD_LABELS[field]:
        if len(synonym) < 5:
            continue
        norm_syn = document._normalize(synonym)
        for row in rows:
            if id(row) in seen:
                continue
            for token in re.findall(r"[a-z]+", document._normalize(row["text"])):
                if len(token) < 5:
                    continue
                if difflib.SequenceMatcher(None, norm_syn, token).ratio() >= 0.82:
                    found.append((row, "label_fuzzy", synonym))
                    seen.add(id(row))
                    break
    return found


def _value_after_label(row_text: str, synonym: str) -> str:
    norm_row = document._normalize(row_text)
    pos = norm_row.find(document._normalize(synonym))
    if pos < 0:
        return ""
    tail = row_text[pos + len(synonym):]
    return tail.lstrip(" :.-–—#\t")


def _row_below(rows: list[dict], row: dict, same_column: bool) -> dict | None:
    candidates = [
        r for r in rows
        if r["page"] == row["page"] and r["box"][1] > row["box"][3] - 2
    ]
    candidates.sort(key=lambda r: r["box"][1])
    for cand in candidates:
        if not same_column:
            return cand
        overlap = min(cand["box"][2], row["box"][2]) - max(cand["box"][0], row["box"][0])
        width = min(cand["box"][2] - cand["box"][0], row["box"][2] - row["box"][0])
        if width > 0 and overlap > 0.5 * width:
            return cand
    return None


def _count_other_labels(text: str, field: str) -> int:
    """Cuántas etiquetas de OTROS campos aparecen en el texto."""
    norm = document._normalize(text)
    found = 0
    for other, synonyms in FIELD_LABELS.items():
        if other == field:
            continue
        if any(len(syn) >= 5 and document._normalize(syn) in norm for syn in synonyms):
            found += 1
    return found


def _is_header_remainder(text: str, field: str) -> bool:
    """True si el texto son encabezados de otras columnas y no un valor.

    Se exige 2+ etiquetas ajenas porque una sola palabra como "Cajas" o
    "Sacos" es un valor legítimo de embalaje aunque también figure entre los
    sinónimos de cantidad.
    """
    return _count_other_labels(text, field) >= 2


def _extract_one(rows: list[dict], field: str, use_semantic: bool) -> dict:
    out = _empty_field()
    label_hits = _find_label_rows(rows, field)
    method = label_hits[0][1] if label_hits else "none"

    searched: list[tuple[dict, str, str]] = []
    for row, _, synonym in label_hits:
        tail = _value_after_label(row["text"], synonym) if synonym else ""
        # Si lo que sigue a la etiqueta son encabezados de otras columnas, el
        # valor real está en la fila de abajo, no al lado (layout de tabla).
        if tail.strip() and not _is_header_remainder(tail, field):
            searched.append((row, tail, "same_row_after_label"))
        below_col = _row_below(rows, row, same_column=True)
        if below_col is not None and not _is_header_remainder(below_col["text"], field):
            searched.append((below_col, below_col["text"], "row_below_same_column"))
        below = _row_below(rows, row, same_column=False)
        if below is not None and not _is_header_remainder(below["text"], field):
            searched.append((below, below["text"], "row_below_full"))

    if not searched and use_semantic:
        embedder = document.get_embedder()
        texts = [r["text"] for r in rows]
        if texts:
            emb = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            q = embedder.encode([SEMANTIC_QUERIES[field]], convert_to_numpy=True, normalize_embeddings=True)[0]
            scores = emb @ q
            # Umbral mínimo: es más honesto decir "no encontrado" que adivinar.
            for idx in np.argsort(-scores)[:3]:
                if scores[int(idx)] < SEMANTIC_MIN_SIMILARITY:
                    continue
                searched.append((rows[int(idx)], rows[int(idx)]["text"], "semantic_row"))
            if searched:
                method = "semantic"

    # Para la cantidad se prefiere una unidad CONTABLE ("18 cajas") sobre un
    # peso o volumen ("90 kg"), porque el peso no se puede verificar contando.
    # Las filas monetarias se descartan por completo: sin esto, "Valor total:
    # USD 3.750,00" se interpretaría como 3750 bultos.
    resolved = None
    fallback = None
    for src_row, text, strategy in searched:
        if field == "cantidad" and (_has_money_hint(text) or _has_money_hint(src_row["text"])):
            continue
        value = _finalize_value(field, text)
        if value is None:
            continue
        if field == "cantidad" and not value["es_contable"]:
            if fallback is None:
                fallback = (src_row, text, strategy, value)
            continue
        resolved = (src_row, text, strategy, value)
        break
    if resolved is None:
        resolved = fallback

    if resolved is not None:
        src_row, text, strategy, value = resolved
        base = METHOD_BASE.get(method if method != "none" else "regex_only", 0.35)
        out.update(value)
        out["confidence"] = round(min(0.99, max(0.05, base * src_row["ocr_confidence"])), 4)
        out["method"] = method if method != "none" else "regex_only"
        out["strategy"] = strategy
        out["source_text"] = src_row["text"]
        out["source_box"] = src_row["box"]
        out["page"] = src_row["page"]
        out["raw"] = text.strip()

    # Candidatos alternativos (para el dropdown de la UI)
    for r in rows:
        if out["source_text"] is not None and r["text"] == out["source_text"]:
            continue
        alt = _finalize_value(field, r["text"])
        if alt and alt.get("value") and alt["value"] != out["value"]:
            out["candidates"].append(
                {
                    "value": alt["value"],
                    "number": alt.get("number"),
                    "confidence": round(0.30 * r["ocr_confidence"], 4),
                    "source_text": r["text"][:90],
                }
            )
    out["candidates"] = out["candidates"][:4]
    return out


def _finalize_value(field: str, text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None

    if field == "cantidad":
        number, unit, countable = parse_quantity(text)
        if number is None:
            return None
        label = f"{number} {unit}".strip() if unit else str(number)
        return {"value": label, "number": number, "unit": unit, "es_contable": countable}

    if field == "vehiculo":
        for m in RE_PLATE.finditer(text):
            prefix = m.group("plate").lower()
            if prefix in PLATE_BLOCKLIST:
                continue
            # Rechazar si está dentro de un token largo (cédula, RUC)
            token = next(
                (t for t in re.findall(r"[A-Za-z0-9\-]+", text) if m.group(0).replace(" ", "") in t.replace(" ", "")),
                "",
            )
            if len(re.sub(r"\D", "", token)) > 5:
                continue
            plate = f"{m.group('plate').upper()}-{m.group('digits')}"
            return {"value": plate, "number": None, "unit": None, "es_contable": False}
        return None

    if field == "orden":
        # Un número de orden real siempre trae al menos un dígito. Exigirlo
        # evita falsos positivos como "ORDEN DE TRANSPORTE" -> "SPORTE".
        m = RE_ORDER.search(text)
        if m and any(c.isdigit() for c in m.group("id")):
            return {"value": m.group("id").upper(), "number": None, "unit": None, "es_contable": False}
        stripped = text.strip(" :.-#")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-/]{2,19}", stripped) and any(c.isdigit() for c in stripped):
            return {"value": stripped.upper(), "number": None, "unit": None, "es_contable": False}
        return None

    # producto / embalaje: texto libre
    cleaned = re.sub(r"\s+", " ", text).strip(" :.-–—")
    if not cleaned or len(cleaned) < 2:
        return None
    return {"value": cleaned[:120], "number": None, "unit": None, "es_contable": False}


def extract_order_fields(lines: list[dict], use_semantic: bool = True) -> dict:
    """Extrae los 5 campos de la orden a partir de las líneas de OCR."""
    rows = group_lines_into_rows(lines)
    fields = {name: _extract_one(rows, name, use_semantic) for name in FIELD_LABELS}

    # El prompt sugerido sale del embalaje; si no hay, se intenta con la
    # unidad de la cantidad ("18 cajas" -> cajas).
    source_text = fields["embalaje"]["value"] or fields["cantidad"]["unit"] or ""
    prompt, prompt_source = map_packaging_term(source_text)

    return {
        "rows": rows,
        "fields": fields,
        "suggested_prompt": prompt,
        "suggested_term_source": prompt_source,
    }


# ---------------------------------------------------------------------------
# Veredicto
# ---------------------------------------------------------------------------

_ORDER = {"coincide": 0, "revision": 1, "no_coincide": 2}
_LABELS = {"coincide": "Coincide", "revision": "Revisión", "no_coincide": "No coincide", "no_aplica": "No aplica"}


def compute_verdict(
    layer1: dict,
    layer2: dict,
    layer3: dict,
    expected: int | None,
    expected_countable: bool = True,
    tolerance: float = 0.15,
    max_countable: int = 30,
) -> dict:
    """Combina las tres capas. El global es la PEOR de las capas aplicables."""
    count = layer1["count"]

    if expected is None or not expected_countable:
        l1_status = "revision"
        l1_reason = "cantidad no contable (peso o volumen); ingrese el número de bultos manualmente"
    elif expected > max_countable:
        l1_status = "revision"
        l1_reason = f"cantidad de {expected} demasiado alta para verificar por conteo en foto"
    elif count == 0:
        # Casi siempre es prompt equivocado o mal ángulo, no un camión vacío.
        l1_status = "revision"
        l1_reason = "no se detectó ninguna unidad; revise el término de embalaje en inglés"
    else:
        diff = abs(count - expected)
        allow = max(1, math.floor(expected * tolerance))
        if diff <= allow:
            l1_status = "coincide"
            l1_reason = f"se detectaron {count} unidades; el documento indica {expected}"
        elif diff <= 2 * allow:
            l1_status = "revision"
            l1_reason = f"se detectaron {count} unidades frente a {expected} declaradas"
        else:
            l1_status = "no_coincide"
            l1_reason = f"se detectaron {count} unidades frente a {expected} declaradas"

    if l1_status == "coincide" and layer1.get("reliability") == "baja":
        l1_status = "revision"
        l1_reason += " (conteo poco confiable)"

    l2_map = {
        "encontrado": "coincide",
        "no_legible": "no_aplica",
        "producto_distinto": "no_coincide",
        "no_aplica": "no_aplica",
    }
    l2_status = l2_map.get(layer2.get("status", "no_aplica"), "no_aplica")
    l3_status = layer3.get("status", "no_aplica")

    applicable = [s for s in (l1_status, l2_status, l3_status) if s in _ORDER]
    global_status = max(applicable, key=lambda s: _ORDER[s]) if applicable else "revision"

    reasons = [l1_reason]
    if l2_status == "coincide":
        reasons.append(f'producto confirmado por etiqueta: "{layer2.get("matched_text")}"')
    elif l2_status == "no_coincide":
        reasons.append("el texto leído en la carga no corresponde al producto declarado")
    elif layer2.get("status") == "no_legible":
        reasons.append("no se pudo leer texto en la carga")
    if l3_status in _ORDER:
        reasons.append(f'similitud con la referencia: {layer3.get("similarity")}')

    return {
        "verdict": global_status,
        "verdict_label": _LABELS[global_status],
        "verdict_reason": ". ".join(r for r in reasons if r).strip() + ".",
        "capa1_status": l1_status,
        "capa1_reason": l1_reason,
        "capa2_status": l2_status,
        "capa3_status": l3_status,
    }
