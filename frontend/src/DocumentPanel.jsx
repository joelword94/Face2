import { useEffect, useState } from "react";
import "./DocumentPanel.css";
// Reutiliza .verdict-badge / .layers / .layer-card, que ya viven ahí.
import "./CargoVerifyPanel.css";

const API_URL = "http://localhost:8000";

const VERDICT_CLASS = {
  coincide: "ok",
  revision: "warn",
  no_coincide: "bad",
};

function pct(v) {
  return `${((v || 0) * 100).toFixed(1)}%`;
}

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

  const [plantillas, setPlantillas] = useState([]);
  const [nuevaPlantilla, setNuevaPlantilla] = useState(null);
  const [nombrePlantilla, setNombrePlantilla] = useState("");
  const [registrando, setRegistrando] = useState(false);

  const [plantillaForzada, setPlantillaForzada] = useState("");
  const [formatoResult, setFormatoResult] = useState(null);
  const [formatoLoading, setFormatoLoading] = useState(false);
  const [verDetalle, setVerDetalle] = useState(false);

  useEffect(() => {
    cargarPlantillas();
  }, []);

  async function cargarPlantillas() {
    try {
      const res = await fetch(`${API_URL}/plantillas`);
      const data = await res.json();
      setPlantillas(data.plantillas || []);
    } catch {
      // El banco vacío no es un error que valga la pena mostrar al abrir.
    }
  }

  function handleFileChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setOcrData(null);
    setExactResults(null);
    setSemanticResults(null);
    setSealResult(null);
    setFormatoResult(null);
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

  async function handleRegistrarPlantilla() {
    if (!nuevaPlantilla || !nombrePlantilla.trim()) return;
    setRegistrando(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", nuevaPlantilla);
    formData.append("nombre", nombrePlantilla);

    try {
      const res = await fetch(`${API_URL}/plantillas`, { method: "POST", body: formData });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setNuevaPlantilla(null);
      setNombrePlantilla("");
      await cargarPlantillas();
    } catch (err) {
      setError(err.message);
    } finally {
      setRegistrando(false);
    }
  }

  async function handleEliminarPlantilla(id) {
    try {
      await fetch(`${API_URL}/plantillas/${id}`, { method: "DELETE" });
      await cargarPlantillas();
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleVerificarFormato() {
    if (!file) return;
    setFormatoLoading(true);
    setFormatoResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);
    if (plantillaForzada) formData.append("plantilla_id", plantillaForzada);
    // Si la sección 1 ya extrajo el texto, se reenvía para no repetir el OCR,
    // que es con diferencia lo más lento del flujo.
    if (ocrData) {
      formData.append("lines", JSON.stringify(ocrData.lines));
      formData.append("paragraphs", JSON.stringify(ocrData.paragraphs));
    }

    try {
      const res = await fetch(`${API_URL}/plantillas/verificar`, { method: "POST", body: formData });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setFormatoResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setFormatoLoading(false);
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

      <section className="doc-section">
        <h2>5. Banco de plantillas</h2>
        <p className="hint">
          Registra una vez cada formato válido (certificado médico, orden de compra...). Se procesa en el momento
          del alta y queda guardado, así que verificar después es rápido. El alta sí tarda: corre el OCR y busca los
          sellos y firmas de la plantilla.
        </p>

        <div className="row">
          <input
            key={plantillas.length}
            type="file"
            accept="image/*,.pdf"
            onChange={(e) => setNuevaPlantilla(e.target.files[0] || null)}
          />
          <input
            type="text"
            value={nombrePlantilla}
            onChange={(e) => setNombrePlantilla(e.target.value)}
            placeholder="Nombre del formato (ej. certificado médico)"
          />
          <button
            onClick={handleRegistrarPlantilla}
            disabled={!nuevaPlantilla || !nombrePlantilla.trim() || registrando}
          >
            {registrando ? "Procesando plantilla..." : "Registrar plantilla"}
          </button>
        </div>

        {plantillas.length === 0 ? (
          <p className="hint">El banco está vacío. Registra al menos un formato para poder verificar documentos.</p>
        ) : (
          <table className="plantillas-tabla">
            <thead>
              <tr>
                <th>Formato</th>
                <th>Páginas</th>
                <th>Bloques</th>
                <th>Sellos / firmas</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {plantillas.map((p) => (
                <tr key={p.id}>
                  <td>{p.nombre}</td>
                  <td>{p.num_pages}</td>
                  <td>{p.num_bloques}</td>
                  <td>{p.num_graficos > 0 ? p.graficos.map((g) => g.label).join(", ") : "—"}</td>
                  <td>
                    <button className="link-danger" onClick={() => handleEliminarPlantilla(p.id)}>
                      Eliminar
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="doc-section">
        <h2>6. Verificar formato contra el banco</h2>
        <p className="hint">
          Compara el documento cargado arriba contra las plantillas y dice a cuál corresponde y qué le falta. Solo se
          exige que el documento contenga lo que la plantilla tiene: los datos rellenados (nombres, cédulas, fechas)
          no penalizan.
        </p>

        <div className="row">
          <select value={plantillaForzada} onChange={(e) => setPlantillaForzada(e.target.value)}>
            <option value="">Buscar la mejor del banco</option>
            {plantillas.map((p) => (
              <option key={p.id} value={p.id}>
                Comparar solo contra: {p.nombre}
              </option>
            ))}
          </select>
          <button onClick={handleVerificarFormato} disabled={!file || plantillas.length === 0 || formatoLoading}>
            {formatoLoading ? "Comparando..." : "Verificar formato"}
          </button>
        </div>

        {!file && <p className="hint">Carga primero un documento en la sección 1.</p>}

        {formatoResult && (
          <div className="formato-result">
            <div className={`verdict-badge ${VERDICT_CLASS[formatoResult.verdict]}`}>
              {formatoResult.verdict_label}
              {formatoResult.mejor && <> — {formatoResult.mejor.nombre}</>}
            </div>
            {formatoResult.razones.map((r, i) => (
              <p className="verdict-reason" key={i}>
                {r}
              </p>
            ))}

            {formatoResult.mejor && (
              <>
                <div className="layers">
                  <div className="layer-card">
                    <h4>Cobertura del formato</h4>
                    <p>
                      <strong>{pct(formatoResult.mejor.cobertura)}</strong> de los bloques
                    </p>
                    <p>
                      {formatoResult.mejor.bloques.encontrados.length} encontrados,{" "}
                      {formatoResult.mejor.bloques.faltantes.length} faltantes
                    </p>
                  </div>
                  <div className="layer-card">
                    <h4>Sellos y firmas</h4>
                    {formatoResult.mejor.graficos.status === "no_aplica" ? (
                      <p>La plantilla no tiene ninguno</p>
                    ) : (
                      <p>
                        <strong>
                          {formatoResult.mejor.graficos_presentes} de {formatoResult.mejor.graficos_total}
                        </strong>{" "}
                        encontrados
                      </p>
                    )}
                  </div>
                  <div className="layer-card">
                    <h4>Similitud de texto</h4>
                    <p>
                      <strong>{pct(formatoResult.mejor.similitud_texto)}</strong>
                    </p>
                  </div>
                  <div className="layer-card">
                    <h4>Parecido visual</h4>
                    <p>
                      <strong>{pct(formatoResult.mejor.similitud_visual)}</strong>
                    </p>
                    <p className="hint">Señal de apoyo, no decide</p>
                  </div>
                </div>

                {formatoResult.mejor.bloques.faltantes.length > 0 && (
                  <div className="results">
                    <h3>Falta en el documento ({formatoResult.mejor.bloques.faltantes.length})</h3>
                    <ul>
                      {formatoResult.mejor.bloques.faltantes.map((b, i) => (
                        <li key={i}>
                          "{b.texto}" <span className="prov">— página {b.page} de la plantilla</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {formatoResult.mejor.graficos.status === "ok" && (
                  <div className="results">
                    <h3>Sellos, firmas y logos</h3>
                    <p className="hint">
                      A la izquierda el de la plantilla, a la derecha lo más parecido que se encontró en el documento.
                      Compara tú mismo: el porcentaje dice si se parecen, no si el sello es auténtico.
                    </p>
                    <div className="graficos-grid">
                      {[...formatoResult.mejor.graficos.presentes, ...formatoResult.mejor.graficos.faltantes].map(
                        (g, i) => (
                          <div className={`grafico-par ${g.similitud >= 0.8 ? "ok" : "bad"}`} key={i}>
                            <h4>
                              {g.label} — {pct(g.similitud)} {g.similitud >= 0.8 ? "" : "(no encontrado)"}
                            </h4>
                            <div className="grafico-imgs">
                              {g.recorte_plantilla && <img src={g.recorte_plantilla} alt="plantilla" />}
                              {g.recorte_documento ? (
                                <img src={g.recorte_documento} alt="documento" />
                              ) : (
                                <span className="prov">sin equivalente en el documento</span>
                              )}
                            </div>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                )}

                <button className="link-toggle" onClick={() => setVerDetalle(!verDetalle)}>
                  {verDetalle ? "Ocultar detalle" : "Ver detalle y ranking del banco"}
                </button>

                {verDetalle && (
                  <div className="results">
                    <h3>Ranking del banco</h3>
                    <ul>
                      {formatoResult.ranking.map((r) => (
                        <li key={r.plantilla_id}>
                          <strong>{r.nombre}</strong> — cobertura {pct(r.cobertura)}, texto {pct(r.similitud_texto)},
                          visual {pct(r.similitud_visual)}
                          {r.graficos_total > 0 && (
                            <>
                              , sellos {r.graficos_presentes}/{r.graficos_total}
                            </>
                          )}
                        </li>
                      ))}
                    </ul>
                    <h3>Bloques encontrados ({formatoResult.mejor.bloques.encontrados.length})</h3>
                    <ul>
                      {formatoResult.mejor.bloques.encontrados.map((b, i) => (
                        <li key={i}>
                          "{b.texto}" <span className="prov">— {pct(b.score)} por vía {b.via}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

export default DocumentPanel;
