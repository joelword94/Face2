import { useState } from "react";
import "./DocumentPanel.css";

const API_URL = "http://localhost:8000";

function DocumentPanel() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [ocrLoading, setOcrLoading] = useState(false);
  const [ocrData, setOcrData] = useState(null);
  const [error, setError] = useState(null);

  const [exactQuery, setExactQuery] = useState("");
  const [exactResults, setExactResults] = useState(null);
  const [exactLoading, setExactLoading] = useState(false);

  const [semanticQuery, setSemanticQuery] = useState("");
  const [semanticResults, setSemanticResults] = useState(null);
  const [semanticLoading, setSemanticLoading] = useState(false);

  const [sealPrompt, setSealPrompt] = useState("stamp. seal.");
  const [sealConf, setSealConf] = useState(0.2);
  const [referenceFile, setReferenceFile] = useState(null);
  const [sealResult, setSealResult] = useState(null);
  const [sealLoading, setSealLoading] = useState(false);

  function handleFileChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setOcrData(null);
    setExactResults(null);
    setSemanticResults(null);
    setSealResult(null);
    setError(null);
  }

  async function handleRunOcr() {
    if (!file) return;
    setOcrLoading(true);
    setError(null);
    setOcrData(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/document/ocr`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Error del servidor: ${res.status}`);
      const data = await res.json();
      setOcrData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setOcrLoading(false);
    }
  }

  async function handleExactSearch() {
    if (!ocrData || !exactQuery) return;
    setExactLoading(true);
    setExactResults(null);

    const formData = new FormData();
    formData.append("lines", JSON.stringify(ocrData.lines));
    formData.append("query", exactQuery);

    try {
      const res = await fetch(`${API_URL}/document/search-exact`, { method: "POST", body: formData });
      const data = await res.json();
      setExactResults(data.matches);
    } catch (err) {
      setError(err.message);
    } finally {
      setExactLoading(false);
    }
  }

  async function handleSemanticSearch() {
    if (!ocrData || !semanticQuery) return;
    setSemanticLoading(true);
    setSemanticResults(null);

    const formData = new FormData();
    formData.append("paragraphs", JSON.stringify(ocrData.paragraphs));
    formData.append("query", semanticQuery);

    try {
      const res = await fetch(`${API_URL}/document/search-semantic`, { method: "POST", body: formData });
      const data = await res.json();
      setSemanticResults(data.matches);
    } catch (err) {
      setError(err.message);
    } finally {
      setSemanticLoading(false);
    }
  }

  async function handleSealDetect() {
    if (!file) return;
    setSealLoading(true);
    setSealResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("prompt", sealPrompt);
    formData.append("conf", sealConf);
    if (referenceFile) formData.append("reference_file", referenceFile);

    try {
      const res = await fetch(`${API_URL}/document/seal`, { method: "POST", body: formData });
      const data = await res.json();
      setSealResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setSealLoading(false);
    }
  }

  return (
    <div className="doc-app">
      <h1>Análisis de documentos</h1>

      <section className="doc-section">
        <h2>1. Cargar documento</h2>
        <input type="file" accept="image/*,.pdf" onChange={handleFileChange} />
        <button onClick={handleRunOcr} disabled={!file || ocrLoading}>
          {ocrLoading ? "Extrayendo texto..." : "Extraer texto (OCR)"}
        </button>

        {error && <p className="error">{error}</p>}

        <div className="doc-preview">
          {preview && file?.type === "application/pdf" && (
            <div className="image-box">
              <h3>Documento (PDF)</h3>
              <iframe title="documento" src={preview} width="400" height="500" />
            </div>
          )}
          {preview && file?.type !== "application/pdf" && (
            <div className="image-box">
              <h3>Documento</h3>
              <img src={preview} alt="documento" />
            </div>
          )}
          {ocrData && (
            <div className="ocr-text-box">
              <h3>
                Texto extraído ({ocrData.num_pages} {ocrData.num_pages === 1 ? "página" : "páginas"},{" "}
                {ocrData.paragraphs.length} párrafos)
              </h3>
              <pre>{ocrData.full_text}</pre>
            </div>
          )}
        </div>
      </section>

      {ocrData && (
        <section className="doc-section">
          <h2>2. Búsqueda de texto exacto</h2>
          <p className="hint">Para nombres, cédulas, fechas o frases escritas tal cual (ej. "JOEL FIGUEROA").</p>
          <div className="row">
            <input
              type="text"
              value={exactQuery}
              onChange={(e) => setExactQuery(e.target.value)}
              placeholder="Texto a buscar..."
            />
            <button onClick={handleExactSearch} disabled={!exactQuery || exactLoading}>
              {exactLoading ? "Buscando..." : "Buscar"}
            </button>
          </div>
          {exactResults && (
            <div className="results">
              {exactResults.length === 0 && <p>No se encontró ese texto en el documento.</p>}
              <ul>
                {exactResults.map((r, i) => (
                  <li key={i}>
                    <strong>"{r.text}"</strong> — página {r.page} — confianza OCR {(r.confidence * 100).toFixed(1)}%
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      {ocrData && (
        <section className="doc-section">
          <h2>3. Búsqueda semántica (por significado)</h2>
          <p className="hint">
            Para conceptos o cláusulas que pueden estar redactadas distinto (ej. "debe usar uniforme" aunque el
            documento diga "portar la indumentaria corporativa").
          </p>
          <div className="row">
            <input
              type="text"
              value={semanticQuery}
              onChange={(e) => setSemanticQuery(e.target.value)}
              placeholder="Describe lo que buscas..."
            />
            <button onClick={handleSemanticSearch} disabled={!semanticQuery || semanticLoading}>
              {semanticLoading ? "Buscando..." : "Buscar"}
            </button>
          </div>
          {semanticResults && (
            <div className="results">
              <ul>
                {semanticResults.map((r, i) => (
                  <li key={i}>
                    <strong>{(r.similarity * 100).toFixed(1)}% similitud</strong> — página {r.page} — "{r.text}"
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}

      <section className="doc-section">
        <h2>4. Detección y comparación de sello / firma / logo</h2>
        <p className="hint">
          Detecta regiones visuales (sello, firma, logo) según el prompt de texto. Si subes una imagen de referencia,
          compara cada detección contra ella y da un porcentaje de similitud.
        </p>

        <div className="seal-controls">
          <label>
            Prompt de texto:
            <input type="text" value={sealPrompt} onChange={(e) => setSealPrompt(e.target.value)} />
          </label>

          <label>
            Confianza mínima: {sealConf}
            <input
              type="range"
              min="0.05"
              max="0.9"
              step="0.05"
              value={sealConf}
              onChange={(e) => setSealConf(parseFloat(e.target.value))}
            />
          </label>

          <label>
            Imagen de referencia del sello (opcional):
            <input type="file" accept="image/*" onChange={(e) => setReferenceFile(e.target.files[0] || null)} />
          </label>

          <button onClick={handleSealDetect} disabled={!file || sealLoading}>
            {sealLoading ? "Analizando..." : "Detectar y comparar"}
          </button>
        </div>

        {sealResult && (
          <div>
            <p className="hint">
              Analizadas {sealResult.num_pages} {sealResult.num_pages === 1 ? "página" : "páginas"} — verde = alta
              similitud con la referencia, rojo = baja similitud o sin referencia.
            </p>
            {sealResult.detections.length === 0 && <p>No se detectó nada con ese prompt / umbral en ninguna página.</p>}
            {sealResult.pages.map((p) => (
              <div className="doc-preview" key={p.page}>
                <div className="image-box">
                  <h3>Página {p.page}</h3>
                  <img src={p.annotated_image} alt={`resultado página ${p.page}`} />
                </div>
                <div className="results">
                  <h3>Detecciones en esta página ({p.detections.length})</h3>
                  <ul>
                    {p.detections.map((d, i) => (
                      <li key={i}>
                        <strong>{d.label}</strong> — confianza {(d.confidence * 100).toFixed(1)}%
                        {d.match_similarity !== undefined && (
                          <> — similitud con referencia {(d.match_similarity * 100).toFixed(1)}%</>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

export default DocumentPanel;
