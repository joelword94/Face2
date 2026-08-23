"""Banco de plantillas y verificación de formato de documentos.

Un documento subido se compara contra las plantillas registradas para responder
"¿esto corresponde a alguno de mis formatos válidos y le falta algo?".

La comparación es DELIBERADAMENTE ASIMÉTRICA: se mide cuánto del formato de la
plantilla aparece en el documento, nunca al revés. Los datos rellenados (nombre,
cédula, fecha) son contenido extra del documento y por lo tanto no penalizan.
Sin esta asimetría todo documento real puntuaría bajo solo porque los datos
cambian, y el porcentaje dejaría de significar algo.
"""

import base64
import difflib
import io
import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import document

#: El banco vive junto al repo, no en el cwd, para que el backend arranque igual
#: desde la raíz o desde backend/.
PLANTILLAS_DIR = Path(__file__).resolve().parent.parent / "plantillas"

#: Términos para localizar los elementos gráficos. Grounding DINO separa frases
#: por punto, igual que en el detector de sellos.
GRAFICOS_PROMPT = "stamp. seal. signature. logo. letterhead."
GRAFICOS_CONF = 0.25

#: Lado máximo al que se reduce una página antes de la detección. Grounding DINO
#: reescala internamente de todas formas; hacerlo antes ahorra tiempo en CPU.
GRAFICOS_MAX_SIDE = 1333

#: Límites de área para el dedupe, ajustados a páginas de documento.
GRAFICOS_MAX_AREA_FRAC = 0.50
GRAFICOS_MIN_AREA_FRAC = 0.0004

#: Un bloque de la plantilla se da por encontrado a partir de aquí.
BLOQUE_MATCH_MIN = 0.62

#: Similitud CLIP mínima para dar por presente un elemento gráfico. Mismo
#: criterio que usa cargo.compare_against_reference.
GRAFICO_MATCH_MIN = 0.80

COBERTURA_COINCIDE = 0.80
COBERTURA_REVISION = 0.55

#: Si la segunda plantilla del ranking queda más cerca que esto, el "match" no
#: es confiable y se manda a revisión.
MARGEN_RANKING_MIN = 0.05

#: Bloques más cortos que esto son ruido de OCR.
MIN_CHARS_BLOQUE = 4

#: Por debajo de esta longitud útil, un bloque solo puede darse por encontrado
#: por coincidencia literal. MiniLM apenas discrimina cadenas cortas: casi
#: cualquier par de textos de dos palabras da coseno alto, y aceptar eso
#: inflaría la cobertura de cualquier documento.
MIN_CHARS_SEMANTICA = 12

#: Líneas de OCR por debajo de esta confianza son ruido.
MIN_CONF_LINEA = 0.30


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _clip_image_embeddings(images: list[Image.Image]) -> np.ndarray:
    """Embeddings CLIP normalizados de una lista de imágenes."""
    if not images:
        return np.zeros((0, 512), dtype="float32")
    model, processor = document.get_clip()
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    if hasattr(feats, "pooler_output"):
        feats = feats.pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


#: Lado máximo de un recorte al mandarlo a la interfaz. Los recortes salen a
#: resolución de página (200 dpi) y pesan cientos de KB cada uno; en pantalla se
#: ven a 180 px. La comparación CLIP no usa esta versión, así que reducirla no
#: afecta a ninguna medida.
RECORTE_MAX_SIDE = 400


def _image_to_b64(image: Image.Image) -> str:
    escala = min(1.0, RECORTE_MAX_SIDE / max(image.size))
    if escala < 1.0:
        image = image.resize((max(1, int(image.width * escala)), max(1, int(image.height * escala))))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


#: Rellenos de formulario: rayas, puntos suspensivos, [CAMPO], {{campo}}, XXXX.
_RELLENO_PATTERNS = [
    re.compile(r"[_\-–—]{2,}"),
    re.compile(r"\.{3,}"),
    re.compile(r"\[[^\]]{0,40}\]"),
    re.compile(r"\{\{[^}]{0,40}\}\}"),
    re.compile(r"\b[xX]{3,}\b"),
]


def _limpiar_relleno(text: str) -> str:
    """Quita los espacios en blanco a rellenar del formulario.

    De "Nombre del paciente: ______" lo que importa es la etiqueta.
    """
    out = text
    for pattern in _RELLENO_PATTERNS:
        out = pattern.sub(" ", out)
    return " ".join(out.split())


_FECHA_RE = re.compile(r"^\d{1,4}[/\-. ]\d{1,2}[/\-. ]\d{1,4}$")


def _es_bloque_de_datos(text: str) -> bool:
    """True si el bloque parece un dato rellenado y no parte del formato.

    Cubre el caso de registrar como plantilla un ejemplar ya lleno en vez de un
    formulario en blanco. Es deliberadamente conservador: ante la duda se
    conserva el bloque, porque descartar formato real hace subir la cobertura de
    cualquier documento y vuelve inútil la verificación.
    """
    limpio = text.strip()
    if not limpio:
        return True

    if _FECHA_RE.match(limpio):
        return True

    alfanum = [c for c in limpio if c.isalnum()]
    if not alfanum:
        return True

    digitos = sum(1 for c in alfanum if c.isdigit())
    if digitos / len(alfanum) > 0.6:
        return True

    # Sin ninguna palabra de más de 3 letras no hay etiqueta que reconocer.
    palabras = re.findall(r"[^\W\d_]+", limpio, flags=re.UNICODE)
    if not any(len(p) > 3 for p in palabras):
        return True

    return False


def _texto_util(text: str) -> str:
    return "".join(c for c in document._normalize(text) if c.isalnum())


def _fuzzy_score(bloque_norm: str, parrafo_norm: str) -> float:
    """Coincidencia literal tolerante entre un bloque y un párrafo.

    La contención vale 1.0 y es el caso que más importa: la plantilla dice
    "Nombre del paciente:" y el documento dice "Nombre del paciente: Juan Pérez".
    Sin esto el ratio de difflib castigaría al documento por traer el dato.
    """
    if not bloque_norm or not parrafo_norm:
        return 0.0
    if bloque_norm in parrafo_norm:
        return 1.0
    matcher = difflib.SequenceMatcher(None, bloque_norm, parrafo_norm)
    if matcher.quick_ratio() < 0.5:
        return matcher.quick_ratio()
    return matcher.ratio()


# ---------------------------------------------------------------------------
# Preparación de una plantilla
# ---------------------------------------------------------------------------

def _indices_de_valor(lines: list[dict]) -> set[int]:
    """Índices de las líneas que son un VALOR rellenado, no parte del formato.

    Usa la única señal fiable que hay en una sola plantilla: la disposición. En
    una fila visual como  "Exportador:" | "AGRICOLA DEL VALLE S.A."  todo lo que
    va a la derecha de una etiqueta terminada en dos puntos es el dato.

    Sin esto, registrar como plantilla un ejemplar ya lleno mete los datos de
    ESE ejemplar entre los bloques exigidos, y cualquier otro documento del
    mismo formato aparece como incompleto solo por traer otro proveedor u otra
    fecha.

    En documentos de prosa (contratos, cartas) no hay etiquetas con dos puntos,
    así que no se descarta nada y la función es inocua.
    """
    import cargo

    valores: set[int] = set()
    for row in cargo.group_lines_into_rows(lines):
        indices = row["line_indices"]
        ultima_etiqueta = None
        for pos, idx in enumerate(indices):
            if lines[idx]["text"].strip().endswith(":"):
                ultima_etiqueta = pos
        if ultima_etiqueta is not None:
            valores.update(indices[ultima_etiqueta + 1:])
    return valores


def _preparar_bloques(lines: list[dict]) -> list[dict]:
    """Deja solo las líneas que representan el FORMATO, ya limpias.

    Se trabaja con LÍNEAS y no con párrafos a propósito. El agrupador de
    document.run_ocr fusiona líneas cuyo hueco vertical es menor que su altura,
    y en un documento denso eso colapsa la página entera en uno o dos párrafos
    donde el texto fijo queda pegado a los datos rellenados. Un bloque así no
    coincide nunca con otro ejemplar del mismo formato, y la cobertura se va a
    cero. A nivel de línea, "Orden de Compra No:" y su número quedan separados,
    que es justo lo que hace falta para distinguir formato de dato.
    """
    valores = _indices_de_valor(lines)
    bloques = []
    for i, line in enumerate(lines):
        if i in valores:
            continue
        if line.get("confidence", 1.0) < MIN_CONF_LINEA:
            continue
        limpio = _limpiar_relleno(line["text"])
        if len(_texto_util(limpio)) < MIN_CHARS_BLOQUE:
            continue
        if _es_bloque_de_datos(limpio):
            continue
        bloques.append(
            {
                "texto": line["text"],
                "texto_limpio": limpio,
                "page": line.get("page", 1),
                "box": line.get("box"),
            }
        )
    return bloques


def _detectar_graficos(pages: list[Image.Image]) -> list[dict]:
    """Localiza sellos, firmas y logos en las páginas y devuelve sus recortes."""
    import cargo

    encontrados = []
    for page_num, page in enumerate(pages, start=1):
        escala = min(1.0, GRAFICOS_MAX_SIDE / max(page.size))
        if escala < 1.0:
            chico = page.resize((int(page.width * escala), int(page.height * escala)))
        else:
            chico = page

        raw = cargo._run_grounding_dino(chico, GRAFICOS_PROMPT, GRAFICOS_CONF, GRAFICOS_CONF)
        # Sin dedupe el mismo sello aparece varias veces y el conteo de
        # elementos deja de significar nada.
        #
        # Los límites de área por defecto son para fotos de carga, no para
        # documentos: un sello ocupa una fracción mínima de una página A4, así
        # que el mínimo se baja para no descartarlo, y el máximo se aprieta
        # porque en una página ningún sello legítimo ocupa media hoja — esa caja
        # es el artefacto de "página completa" típico de Grounding DINO.
        kept, _ = cargo.dedupe_detections(
            raw,
            chico.size,
            max_area_frac=GRAFICOS_MAX_AREA_FRAC,
            min_area_frac=GRAFICOS_MIN_AREA_FRAC,
        )

        # dedupe_detections trae dos redes de seguridad que conservan la mejor
        # caja aunque el filtro de área las descarte todas. Para contar bultos
        # eso es correcto (una foto de carga siempre tiene al menos uno), pero
        # aquí es justo lo contrario: una página SIN sello debe dar cero. Sin
        # este recorte, Grounding DINO devuelve su caja de "página completa" y
        # se compara página contra página, dando ~0.89 de similitud entre dos
        # documentos que no tienen sello ninguno.
        area_pagina = float(chico.size[0] * chico.size[1]) or 1.0
        kept = [
            d
            for d in kept
            if GRAFICOS_MIN_AREA_FRAC
            <= ((d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])) / area_pagina
            <= GRAFICOS_MAX_AREA_FRAC
        ]

        for d in kept:
            x1, y1, x2, y2 = [v / escala for v in d["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(page.width, x2), min(page.height, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            encontrados.append(
                {
                    "label": d["label"],
                    "confidence": d["confidence"],
                    "page": page_num,
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "crop": page.crop((x1, y1, x2, y2)),
                }
            )
    return encontrados


def registrar_plantilla(file_bytes: bytes, filename: str, nombre: str) -> dict:
    """Da de alta una plantilla: OCR, bloques, elementos gráficos y cachés.

    Todo el trabajo caro se hace aquí, una sola vez. Después verificar contra
    N plantillas cuesta un OCR (el del documento) más N multiplicaciones de
    matrices.
    """
    PLANTILLAS_DIR.mkdir(parents=True, exist_ok=True)

    plantilla_id = uuid.uuid4().hex[:12]
    carpeta = PLANTILLAS_DIR / plantilla_id
    (carpeta / "graficos").mkdir(parents=True, exist_ok=True)

    try:
        extension = Path(filename).suffix.lower() or ".pdf"
        (carpeta / f"original{extension}").write_bytes(file_bytes)

        pages = document.load_document_pages(file_bytes, filename)
        _, lines, paragraphs = document.run_ocr(pages)
        bloques = _preparar_bloques(lines)

        embedder = document.get_embedder()
        if bloques:
            bloque_emb = embedder.encode(
                [b["texto_limpio"] for b in bloques],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
        else:
            bloque_emb = np.zeros((0, 384), dtype="float32")
        np.save(carpeta / "bloques.npy", bloque_emb)

        graficos = _detectar_graficos(pages)
        graficos_meta = []
        for i, g in enumerate(graficos):
            ruta = f"graficos/{i}.png"
            g["crop"].save(carpeta / ruta)
            graficos_meta.append(
                {
                    "label": g["label"],
                    "confidence": g["confidence"],
                    "page": g["page"],
                    "box": g["box"],
                    "archivo": ruta,
                }
            )
        graficos_emb = _clip_image_embeddings([g["crop"] for g in graficos])
        np.save(carpeta / "graficos.npy", graficos_emb)

        visual_emb = _clip_image_embeddings([pages[0]])
        np.save(carpeta / "visual.npy", visual_emb)

        meta = {
            "id": plantilla_id,
            "nombre": nombre.strip() or f"plantilla {plantilla_id}",
            "archivo": f"original{extension}",
            "creado": datetime.now().isoformat(timespec="seconds"),
            "num_pages": len(pages),
            "num_lineas": len(lines),
            "num_parrafos": len(paragraphs),
            "bloques": bloques,
            "graficos": graficos_meta,
        }
        (carpeta / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(carpeta, ignore_errors=True)
        raise

    return _resumen(meta)


def _resumen(meta: dict) -> dict:
    return {
        "id": meta["id"],
        "nombre": meta["nombre"],
        "creado": meta["creado"],
        "num_pages": meta["num_pages"],
        "num_bloques": len(meta["bloques"]),
        "num_graficos": len(meta["graficos"]),
        "graficos": [{"label": g["label"], "page": g["page"]} for g in meta["graficos"]],
    }


def _cargar_meta(carpeta: Path) -> dict | None:
    ruta = carpeta / "meta.json"
    if not ruta.exists():
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def listar_plantillas() -> list[dict]:
    if not PLANTILLAS_DIR.exists():
        return []
    resumenes = []
    for carpeta in sorted(PLANTILLAS_DIR.iterdir()):
        if not carpeta.is_dir():
            continue
        meta = _cargar_meta(carpeta)
        if meta:
            resumenes.append(_resumen(meta))
    resumenes.sort(key=lambda r: r["creado"], reverse=True)
    return resumenes


def eliminar_plantilla(plantilla_id: str) -> bool:
    carpeta = PLANTILLAS_DIR / plantilla_id
    # El id viene de la URL: se comprueba que no escape del banco.
    if not carpeta.is_dir() or carpeta.parent != PLANTILLAS_DIR:
        return False
    shutil.rmtree(carpeta, ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# Comparación
# ---------------------------------------------------------------------------

def _comparar_bloques(meta: dict, carpeta: Path, doc_norm: list[str], doc_emb: np.ndarray) -> dict:
    bloques = meta["bloques"]
    if not bloques:
        return {
            "cobertura": 0.0,
            "similitud_texto": 0.0,
            "encontrados": [],
            "faltantes": [],
            "status": "no_aplica",
        }

    bloque_emb = np.load(carpeta / "bloques.npy")
    if len(doc_emb) and len(bloque_emb):
        cos = bloque_emb @ doc_emb.T
    else:
        cos = np.zeros((len(bloques), max(len(doc_norm), 1)), dtype="float32")

    encontrados, faltantes, scores = [], [], []
    for i, bloque in enumerate(bloques):
        bloque_norm = document._normalize(bloque["texto_limpio"])
        # En bloques cortos el coseno no discrimina, así que se exige literal.
        solo_literal = len(_texto_util(bloque["texto_limpio"])) < MIN_CHARS_SEMANTICA
        mejor, mejor_j, via = 0.0, None, None
        for j, parrafo_norm in enumerate(doc_norm):
            s_cos = 0.0 if solo_literal else (float(cos[i][j]) if cos.shape[1] > j else 0.0)
            s_fuzzy = _fuzzy_score(bloque_norm, parrafo_norm)
            # El máximo de las dos: el coseno pilla la cláusula redactada
            # distinto, el difuso pilla la etiqueta corta y literal, donde
            # MiniLM apenas discrimina porque todo texto corto se parece.
            s = max(s_cos, s_fuzzy)
            if s > mejor:
                mejor, mejor_j, via = s, j, ("literal" if s_fuzzy >= s_cos else "semantica")

        scores.append(mejor)
        entrada = {
            "texto": bloque["texto_limpio"],
            "page": bloque["page"],
            "score": round(mejor, 4),
            "via": via,
            "match": doc_norm[mejor_j][:160] if mejor_j is not None else None,
        }
        (encontrados if mejor >= BLOQUE_MATCH_MIN else faltantes).append(entrada)

    return {
        "cobertura": round(len(encontrados) / len(bloques), 4),
        "similitud_texto": round(float(np.mean(scores)), 4),
        "encontrados": encontrados,
        "faltantes": faltantes,
        "status": "ok",
    }


def _comparar_graficos(meta: dict, carpeta: Path, doc_graficos: list[dict], doc_emb: np.ndarray) -> dict:
    graficos = meta["graficos"]
    if not graficos:
        return {"status": "no_aplica", "presentes": [], "faltantes": [], "total": 0}

    plantilla_emb = np.load(carpeta / "graficos.npy")
    presentes, faltantes = [], []

    for i, g in enumerate(graficos):
        # Aquí solo se anota de dónde saldrá cada recorte. Convertirlos a
        # base64 para todas las plantillas del banco sería tirar imágenes que
        # nadie va a ver: solo se adjuntan las de la ganadora.
        entrada = {
            "label": g["label"],
            "page": g["page"],
            "similitud": 0.0,
            "recorte_plantilla": None,
            "recorte_documento": None,
            "_archivo": g["archivo"],
            "_doc_idx": None,
        }

        if len(doc_emb) and i < len(plantilla_emb):
            sims = plantilla_emb[i] @ doc_emb.T
            mejor_j = int(np.argmax(sims))
            entrada["similitud"] = round(float(sims[mejor_j]), 4)
            entrada["label_documento"] = doc_graficos[mejor_j]["label"]
            entrada["page_documento"] = doc_graficos[mejor_j]["page"]
            entrada["_doc_idx"] = mejor_j

        (presentes if entrada["similitud"] >= GRAFICO_MATCH_MIN else faltantes).append(entrada)

    return {
        "status": "ok",
        "presentes": presentes,
        "faltantes": faltantes,
        "total": len(graficos),
    }


def _adjuntar_recortes(graficos: dict, carpeta: Path, doc_graficos: list[dict]) -> None:
    """Rellena los recortes en base64 de la plantilla ganadora, y solo de ella."""
    for entrada in graficos["presentes"] + graficos["faltantes"]:
        ruta = carpeta / entrada.pop("_archivo", "")
        if ruta.is_file():
            entrada["recorte_plantilla"] = _image_to_b64(Image.open(ruta))
        idx = entrada.pop("_doc_idx", None)
        if idx is not None and idx < len(doc_graficos):
            entrada["recorte_documento"] = _image_to_b64(doc_graficos[idx]["crop"])


def compute_verdict_formato(bloques: dict, graficos: dict, margen: float | None) -> dict:
    """Veredicto con el mismo vocabulario que cargo.compute_verdict."""
    razones = []
    cobertura = bloques["cobertura"]

    if bloques["status"] == "no_aplica":
        return {
            "verdict": "revision",
            "verdict_label": "Requiere revisión",
            "razones": ["la plantilla no tiene bloques de formato utilizables"],
        }

    if cobertura >= COBERTURA_COINCIDE:
        verdict = "coincide"
        razones.append(f"el documento cubre {cobertura:.0%} del formato de la plantilla")
    elif cobertura >= COBERTURA_REVISION:
        verdict = "revision"
        razones.append(f"solo cubre {cobertura:.0%} del formato; faltan {len(bloques['faltantes'])} bloques")
    else:
        verdict = "no_coincide"
        razones.append(f"solo cubre {cobertura:.0%} del formato de la plantilla")

    # Un elemento gráfico ausente baja a revisión, NO a no_coincide: localizarlo
    # con Grounding DINO y compararlo con CLIP es bastante más ruidoso que
    # comparar texto, y marcar como inválido un documento legítimo es peor que
    # mandarlo a que lo mire una persona.
    if graficos["status"] == "ok" and graficos["faltantes"]:
        etiquetas = ", ".join(g["label"] for g in graficos["faltantes"])
        razones.append(f"no se encontró en el documento: {etiquetas}")
        if verdict == "coincide":
            verdict = "revision"

    if margen is not None and margen < MARGEN_RANKING_MIN and verdict == "coincide":
        verdict = "revision"
        razones.append("otra plantilla del banco puntúa casi igual; conviene confirmar cuál corresponde")

    etiquetas = {
        "coincide": "Coincide con la plantilla",
        "revision": "Requiere revisión",
        "no_coincide": "No coincide",
    }
    return {"verdict": verdict, "verdict_label": etiquetas[verdict], "razones": razones}


def comparar_documento(
    file_bytes: bytes,
    filename: str,
    lines: list[dict] | None = None,
    paragraphs: list[dict] | None = None,
    plantilla_id: str | None = None,
) -> dict:
    """Compara un documento contra el banco y devuelve la mejor coincidencia."""
    carpetas = []
    if plantilla_id:
        carpeta = PLANTILLAS_DIR / plantilla_id
        if carpeta.is_dir():
            carpetas = [carpeta]
    elif PLANTILLAS_DIR.exists():
        carpetas = [c for c in sorted(PLANTILLAS_DIR.iterdir()) if c.is_dir()]

    metas = []
    for carpeta in carpetas:
        meta = _cargar_meta(carpeta)
        if meta:
            metas.append((carpeta, meta))

    if not metas:
        return {
            "verdict": "no_coincide",
            "verdict_label": "No coincide",
            "razones": ["el banco de plantillas está vacío" if not plantilla_id else "no existe esa plantilla"],
            "mejor": None,
            "ranking": [],
        }

    pages = document.load_document_pages(file_bytes, filename)
    if lines is None or paragraphs is None:
        _, lines, paragraphs = document.run_ocr(pages)

    # Se busca en líneas Y en párrafos: así un bloque encaja sin importar cómo
    # el OCR haya agrupado ESE documento en concreto. Si el documento fusionó
    # media página en un párrafo, la etiqueta sigue estando contenida ahí; si
    # la dejó suelta, está en su línea. dict.fromkeys quita los duplicados
    # conservando el orden.
    doc_textos = list(dict.fromkeys([l["text"] for l in lines] + [p["text"] for p in paragraphs]))
    doc_norm = [document._normalize(t) for t in doc_textos]
    embedder = document.get_embedder()
    if doc_textos:
        doc_emb = embedder.encode(doc_textos, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    else:
        doc_emb = np.zeros((0, 384), dtype="float32")

    doc_visual = _clip_image_embeddings([pages[0]])

    # La detección de elementos gráficos del documento se hace UNA vez y se
    # reutiliza para todo el banco; y solo si alguna plantilla los necesita.
    doc_graficos, doc_graficos_emb = [], np.zeros((0, 512), dtype="float32")
    if any(meta["graficos"] for _, meta in metas):
        doc_graficos = _detectar_graficos(pages)
        doc_graficos_emb = _clip_image_embeddings([g["crop"] for g in doc_graficos])

    resultados = []
    for carpeta, meta in metas:
        bloques = _comparar_bloques(meta, carpeta, doc_norm, doc_emb)
        graficos = _comparar_graficos(meta, carpeta, doc_graficos, doc_graficos_emb)

        visual = 0.0
        ruta_visual = carpeta / "visual.npy"
        if ruta_visual.exists() and len(doc_visual):
            visual = round(float(np.load(ruta_visual)[0] @ doc_visual[0]), 4)

        resultados.append(
            {
                "carpeta": carpeta,
                "plantilla_id": meta["id"],
                "nombre": meta["nombre"],
                "cobertura": bloques["cobertura"],
                "similitud_texto": bloques["similitud_texto"],
                "similitud_visual": visual,
                "graficos_presentes": len(graficos["presentes"]),
                "graficos_total": graficos["total"],
                "bloques": bloques,
                "graficos": graficos,
            }
        )

    resultados.sort(key=lambda r: r["similitud_texto"], reverse=True)
    mejor = resultados[0]
    _adjuntar_recortes(mejor["graficos"], mejor["carpeta"], doc_graficos)
    for r in resultados:
        r.pop("carpeta")  # Path, no serializable
    margen = None
    if len(resultados) > 1:
        margen = round(mejor["similitud_texto"] - resultados[1]["similitud_texto"], 4)

    veredicto = compute_verdict_formato(mejor["bloques"], mejor["graficos"], margen)

    return {
        **veredicto,
        "margen": margen,
        "num_pages": len(pages),
        "num_lineas": len(lines),
        "graficos_detectados_documento": len(doc_graficos),
        "mejor": mejor,
        "ranking": [
            {
                "plantilla_id": r["plantilla_id"],
                "nombre": r["nombre"],
                "cobertura": r["cobertura"],
                "similitud_texto": r["similitud_texto"],
                "similitud_visual": r["similitud_visual"],
                "graficos_presentes": r["graficos_presentes"],
                "graficos_total": r["graficos_total"],
            }
            for r in resultados
        ],
    }
