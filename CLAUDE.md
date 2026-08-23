# entrenado — verificación de carga y documentos por visión

App local de dos piezas: un backend FastAPI con modelos de visión/OCR y un frontend
React (Vite). Todo corre en la máquina, sin servicios externos.

## Levantar el proyecto

Requisitos: Python 3.12 y Node 20+.

**Primera vez en una máquina nueva** (instala dependencias):

```bash
python -m venv .venv && .venv/Scripts/activate && pip install -r backend/requirements.txt && npm --prefix frontend install
```

En Linux/macOS el activate es `source .venv/bin/activate`.

**Arrancar** (dos servidores, ambos deben quedar corriendo):

- Backend: config `backend-api` de `.claude/launch.json` → `python -m uvicorn main:app --reload --port 8000 --app-dir backend`
- Frontend: config `frontend-vite` → `npm --prefix frontend run dev`

Usa `preview_start` con esos nombres, no `Bash`. El frontend abre en
http://localhost:5173 y llama al backend en http://localhost:8000.

**El puerto 8000 es obligatorio**: el frontend lo tiene fijo en cada panel
(`const API_URL = "http://localhost:8000"`). Por eso `backend-api` lleva
`autoPort: false`. Si el puerto está ocupado, libéralo en vez de cambiarlo.

## Detalles que importan al arrancar

- **La primera petición de cada modelo tarda mucho** (minutos). Los pesos de
  Grounding DINO, CLIP, el embedder y EasyOCR se descargan de HuggingFace al
  primer uso y se cachean en `~/.cache`. Hace falta internet la primera vez.
  Las peticiones siguientes son rápidas. No lo interpretes como que se colgó.
- Los YOLO sí están en el repo, en `modelos/`, y cargan al instante.
- `backend/main.py` resuelve `modelos/` relativo al propio archivo, así que el
  backend arranca igual desde la raíz o desde `backend/`.
- Sin GPU todo corre en CPU y funciona; solo es más lento. Para GPU NVIDIA
  instala la rueda CUDA de torch antes del `requirements.txt`.

## Estructura

- `backend/main.py` — endpoints FastAPI y carga perezosa de modelos.
  - `/predict` — detección con YOLO (clases fijas) o Grounding DINO (texto libre)
  - `/document/ocr`, `/document/search-exact`, `/document/search-semantic`, `/document/seal`
  - `/cargo/extract`, `/cargo/verify`
- `backend/document.py` — OCR (EasyOCR), PDF (pymupdf), búsqueda semántica, sellos.
- `backend/cargo.py` — verificación de carga contra orden de compra, en capas:
  conteo de bultos, identidad del producto por etiqueta impresa, y cotejo con
  los campos extraídos del documento.
- `frontend/src/` — un panel por pestaña del menú lateral: `ContainerPanel`
  (probador de detección), `CargoVerifyPanel` (Container 2, la verificación
  completa), `DocumentPanel`.
- `pruebas/` — imágenes y PDFs de prueba. Úsalos para verificar sin pedirle
  archivos al usuario.

## Verificar que quedó bien

```bash
curl -s http://localhost:8000/models
```

Debe responder los tres modelos. Para una prueba de punta a punta:

```bash
curl -s -X POST http://localhost:8000/predict -F "file=@pruebas/contenedor_test1.jpg" -F "model=yolov8s-oiv7" -F "conf=0.3"
```

## Notas sobre Grounding DINO

Al ajustar prompts o umbrales, ten presente cómo puntúa este modelo:

- Hace *grounding por token*, no por frase. Con `"tanker truck"` una caja puede
  activarse solo por `truck` y marcar cualquier camión. Revisa la etiqueta
  devuelta: si es parcial, esa es la causa.
- El score es similitud contrastiva texto-imagen, no probabilidad calibrada.
  Una detección correcta suele caer en 0.4–0.7; arriba de 0.8 es raro. No uses
  la intuición de umbrales de YOLO.
- `/predict` devuelve la salida cruda, sin deduplicar. El flujo de carga sí
  aplica NMS y supresión de cajas anidadas en `cargo.dedupe_detections`.
