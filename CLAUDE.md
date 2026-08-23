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
  - `/plantillas` (GET listar, POST registrar), `/plantillas/{id}` (DELETE),
    `/plantillas/verificar`
- `backend/document.py` — OCR (EasyOCR), PDF (pymupdf), búsqueda semántica, sellos.
- `backend/plantillas.py` — banco de formatos válidos y verificación de un
  documento contra él. Ver la sección de abajo.
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

## Banco de plantillas (`backend/plantillas.py`)

Guarda formatos válidos y verifica si un documento subido corresponde a alguno.
Vive en `plantillas/<id>/`, **fuera de git**: es data del servidor, no del
proyecto. Al clonar en otra máquina el banco arranca vacío.

Lo que hay que entender antes de tocarlo:

- **La comparación es asimétrica a propósito.** Se mide cuánto del formato de la
  plantilla aparece en el documento, nunca al revés. Los datos rellenados
  (nombre, cédula, fecha) son contenido extra del documento y no penalizan. Si
  alguien la vuelve simétrica, todo documento real pasa a puntuar bajo solo
  porque los datos cambian.
- **Los bloques son LÍNEAS de OCR, no párrafos.** El agrupador de
  `document.run_ocr` fusiona líneas cuyo hueco vertical es menor que su altura,
  y en un documento denso colapsa la página entera en uno o dos párrafos con las
  etiquetas pegadas a los datos. Con párrafos la cobertura da 0% siempre; con
  líneas, `orden_A.pdf` produce 11 bloques que son exactamente sus etiquetas.
- **Los valores rellenados se descartan por disposición**, con
  `cargo.group_lines_into_rows`: en una fila `Exportador:` + `AGRICOLA S.A.`,
  lo que va a la derecha de una etiqueta terminada en dos puntos es dato. Es la
  única señal fiable que hay teniendo una sola plantilla. En documentos de prosa
  no hay etiquetas con dos puntos y la regla no descarta nada.
- **Todo el trabajo caro se hace al registrar la plantilla**, no al verificar:
  OCR, embeddings de los bloques, detección de sellos y sus recortes CLIP.
  Verificar contra N plantillas cuesta un OCR más N multiplicaciones de
  matrices.
- **Cada bloque se compara por dos vías y se toma la mayor**: coseno del
  embedding y coincidencia difusa con `difflib`. La difusa trata la contención
  como 1.0, que es el caso clave — la plantilla dice `"Nombre:"` y el documento
  dice `"Nombre: Juan Pérez"`.
- **La capa de sellos solo corre si la plantilla tiene alguno.** Es lo que
  evita pagar una inferencia de Grounding DINO por documento cuando el formato
  no lleva sellos.
- **`_detectar_graficos` vuelve a filtrar por área después de
  `cargo.dedupe_detections`, y hay que dejarlo.** Ese dedupe tiene dos redes de
  seguridad que conservan la mejor caja aunque el filtro de área las descarte
  todas, porque para contar bultos el conteo nunca debe caer a cero. En
  documentos pasa lo contrario: una página sin sello debe dar cero. Sin ese
  filtro extra, Grounding DINO devuelve su caja de "página completa" etiquetada
  como `stamp`, y comparar página contra página da ~0.89 de similitud entre dos
  documentos que no tienen ningún sello. Comprobado: con el filtro, `orden_A` y
  `orden_B` dan 0 sellos y `contrato_3paginas` da 4.
- **Un sello faltante baja a `revision`, no a `no_coincide`.** Localizarlo con
  Grounding DINO y compararlo con CLIP es mucho más ruidoso que comparar texto.
  CLIP responde "hay algo parecido acá", no "este sello es auténtico"; no sirve
  para peritaje.
- Los umbrales son constantes con nombre al inicio del módulo y están sin
  calibrar contra documentos reales.

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
