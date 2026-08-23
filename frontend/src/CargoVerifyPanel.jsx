import { useState } from "react";
import "./DocumentPanel.css";
import "./CargoVerifyPanel.css";

const API_URL = "http://localhost:8000";

const FIELD_ORDER = [
  { key: "producto", label: "Producto" },
  { key: "embalaje", label: "Embalaje" },
  { key: "cantidad", label: "Cantidad" },
  { key: "vehiculo", label: "Vehículo / placa" },
  { key: "orden", label: "N° de orden" },
];

const VERDICT_CLASS = {
  coincide: "ok",
  revision: "warn",
  no_coincide: "bad",
  no_aplica: "na",
};

const VERDICT_ICON = {
  coincide: "✓",
  revision: "⚠",
  no_coincide: "✕",
  no_aplica: "–",
};

function confClass(c) {
  if (c >= 0.75) return "conf-hi";
  if (c >= 0.45) return "conf-mid";
  return "conf-lo";
}

function CargoVerifyPanel() {
  const [docFile, setDocFile] = useState(null);
  const [extracting, setExtracting] = useState(false);
  const [extracted, setExtracted] = useState(null);
  const [fields, setFields] = useState(null);
  const [promptTerm, setPromptTerm] = useState("");
  const [promptSource, setPromptSource] = useState(null);

  const [photoFile, setPhotoFile] = useState(null);
  const [photoPreview, setPhotoPreview] = useState(null);
  const [referenceFile, setReferenceFile] = useState(null);
  const [checkLabel, setCheckLabel] = useState(true);
  const [boxThreshold, setBoxThreshold] = useState(0.3);
  const [iou, setIou] = useState(0.55);
  const [tolerance, setTolerance] = useState(0.15);

  const [verifying, setVerifying] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  function handleDocChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    setDocFile(f);
    setExtracted(null);
    setFields(null);
    setResult(null);
    setError(null);
  }

  function handlePhotoChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    setPhotoFile(f);
    setPhotoPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }

  async function handleExtract() {
    if (!docFile) return;
    setExtracting(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", docFile);

    try {
      const res = await fetch(`${API_URL}/cargo/extract`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Error del servidor: ${res.status}`);
      const data = await res.json();
      setExtracted(data);
      const editable = {};
      for (const { key } of FIELD_ORDER) editable[key] = data.fields[key]?.value ?? "";
      setFields(editable);
      setPromptTerm(data.suggested_prompt || "");
      setPromptSource(data.suggested_term_source);
    } catch (err) {
      setError(err.message);
    } finally {
      setExtracting(false);
    }
  }

  async function handleVerify() {
    if (!photoFile) return;
    setVerifying(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", photoFile);
    formData.append("producto", fields?.producto || "");
    formData.append("embalaje", fields?.embalaje || "");
    formData.append("cantidad", fields?.cantidad || "");
    formData.append("prompt_override", promptTerm || "");
    formData.append("check_label", checkLabel);
    formData.append("box_threshold", boxThreshold);
    formData.append("iou", iou);
    formData.append("tolerance", tolerance);
    if (referenceFile) formData.append("reference_file", referenceFile);

    try {
      const res = await fetch(`${API_URL}/cargo/verify`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Error del servidor: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setVerifying(false);
    }
  }

  const cantidadField = extracted?.fields?.cantidad;
  const noContable = cantidadField && cantidadField.number !== null && !cantidadField.es_contable;

  return (
    <div className="doc-app">
      <h1>Verificación de carga contra orden</h1>
      <p className="hint">
        El guardia fotografía la orden y luego el cargamento. El sistema cuenta la <strong>unidad de carga</strong>{" "}
        (cajas, sacos, pallets) — no el producto, porque el producto viene dentro del embalaje y no es visible.
      </p>

      {error && <p className="error">{error}</p>}

      {/* ---------------------------------------------- Paso 1 */}
      <section className="doc-section">
        <h2>Paso 1 · Foto de la orden</h2>
        <div className="row">
          <input type="file" accept="image/*,application/pdf" onChange={handleDocChange} />
          <button onClick={handleExtract} disabled={!docFile || extracting}>
            {extracting ? "Leyendo documento..." : "Extraer campos"}
          </button>
        </div>
        {extracted && (
          <p className="hint">
            {extracted.num_pages} {extracted.num_pages === 1 ? "página" : "páginas"} · {extracted.rows.length} filas de
            texto reconstruidas
          </p>
        )}
      </section>

      {/* ---------------------------------------------- Paso 2 */}
      {fields && (
        <section className="doc-section">
          <h2>Paso 2 · Campos extraídos (editables)</h2>
          <p className="hint">
            Corrija lo que el OCR haya leído mal antes de verificar. Los chips indican la confianza de cada campo.
          </p>

          <div className="field-grid">
            {FIELD_ORDER.map(({ key, label }) => {
              const meta = extracted.fields[key] || {};
              return (
                <div className="field-row" key={key}>
                  <label htmlFor={`f-${key}`}>{label}</label>
                  <div className="field-input">
                    <input
                      id={`f-${key}`}
                      type="text"
                      value={fields[key]}
                      onChange={(e) => setFields({ ...fields, [key]: e.target.value })}
                      placeholder="(no detectado)"
                    />
                    {meta.candidates?.length > 0 && (
                      <select
                        value=""
                        onChange={(e) => e.target.value && setFields({ ...fields, [key]: e.target.value })}
                      >
                        <option value="">alternativas…</option>
                        {meta.candidates.map((c, i) => (
                          <option key={i} value={c.value}>
                            {c.value}
                          </option>
                        ))}
                      </select>
                    )}
                    <div className="prov">
                      {meta.method && meta.method !== "none"
                        ? `${meta.method}${meta.strategy ? " · " + meta.strategy : ""} · "${(meta.source_text || "").slice(0, 60)}"`
                        : "no encontrado en el documento"}
                    </div>
                  </div>
                  <span className={`conf-chip ${confClass(meta.confidence || 0)}`}>
                    {Math.round((meta.confidence || 0) * 100)}%
                  </span>
                </div>
              );
            })}
          </div>

          {noContable && (
            <p className="warn-box">
              ⚠ "{cantidadField.value}" está en {cantidadField.unit} y <strong>no se puede contar</strong> en una foto.
              Escriba en Cantidad cuántos bultos vienen (ej. "18 cajas").
            </p>
          )}

          <label className={`prompt-override ${promptSource === "fallback" ? "is-fallback" : ""}`}>
            Término de búsqueda en inglés (lo que el detector va a buscar):
            <input
              type="text"
              value={promptTerm}
              onChange={(e) => setPromptTerm(e.target.value)}
              autoFocus={promptSource === "fallback"}
            />
            {promptSource === "fallback" ? (
              <span className="prov">⚠ No se reconoció el embalaje. Verifique o escriba el término en inglés.</span>
            ) : (
              <span className="prov">origen: {promptSource}</span>
            )}
          </label>
        </section>
      )}

      {/* ---------------------------------------------- Paso 3 */}
      {fields && (
        <section className="doc-section">
          <h2>Paso 3 · Foto del cargamento</h2>
          <p className="hint">
            El conteo automático es orientativo y solo ve la cara frontal de la estiba. Verifique siempre la imagen
            anotada.
          </p>

          <div className="seal-controls">
            <label>
              Foto de la carga:
              <input type="file" accept="image/*,application/pdf" onChange={handlePhotoChange} />
            </label>
            <label>
              Foto de referencia (opcional, activa Capa 3):
              <input type="file" accept="image/*" onChange={(e) => setReferenceFile(e.target.files[0] || null)} />
            </label>
            <label className="check">
              <input type="checkbox" checked={checkLabel} onChange={(e) => setCheckLabel(e.target.checked)} />
              Leer etiquetas de la carga (Capa 2)
            </label>
          </div>

          <details className="thresholds">
            <summary>Ajustes avanzados de detección</summary>
            <div className="seal-controls">
              <label>
                Confianza mínima: {boxThreshold}
                <input
                  type="range" min="0.10" max="0.70" step="0.05"
                  value={boxThreshold}
                  onChange={(e) => setBoxThreshold(parseFloat(e.target.value))}
                />
              </label>
              <label>
                IoU (solapamiento): {iou}
                <input
                  type="range" min="0.30" max="0.90" step="0.05"
                  value={iou}
                  onChange={(e) => setIou(parseFloat(e.target.value))}
                />
              </label>
              <label>
                Tolerancia de cantidad: {Math.round(tolerance * 100)}%
                <input
                  type="range" min="0" max="0.5" step="0.05"
                  value={tolerance}
                  onChange={(e) => setTolerance(parseFloat(e.target.value))}
                />
              </label>
            </div>
          </details>

          <button onClick={handleVerify} disabled={!photoFile || verifying}>
            {verifying ? "Verificando..." : "Verificar carga"}
          </button>

          {photoPreview && !result && (
            <div className="image-box">
              <img src={photoPreview} alt="carga" />
            </div>
          )}
        </section>
      )}

      {/* ---------------------------------------------- Resultado */}
      {result && (
        <section className="doc-section">
          <div className={`verdict-badge ${VERDICT_CLASS[result.verdict]}`}>
            {VERDICT_ICON[result.verdict]} {result.verdict_label}
          </div>
          <p className="verdict-reason">{result.verdict_reason}</p>

          <div className="layers">
            <div className={`layer-card ${VERDICT_CLASS[result.capa1_status]}`}>
              <h4>{VERDICT_ICON[result.capa1_status]} Capa 1 · Conteo</h4>
              <p className="count-summary">
                Detectados <strong>{result.capa1.count}</strong>
                {result.expected !== null && <> · esperados <strong>{result.expected}</strong></>}
                {result.capa1.raw_count !== result.capa1.count && (
                  <> · {result.capa1.raw_count - result.capa1.count} descartadas por solapamiento</>
                )}
              </p>
              <p className="prov">
                término: "{result.capa1.prompt_used}" ({result.capa1.prompt_source}) · confiabilidad{" "}
                {result.capa1.reliability}
              </p>
            </div>

            <div className={`layer-card ${VERDICT_CLASS[result.capa2_status]}`}>
              <h4>{VERDICT_ICON[result.capa2_status]} Capa 2 · Etiqueta</h4>
              {result.capa2.status === "encontrado" && <p>Leído: "{result.capa2.matched_text}"</p>}
              {result.capa2.status === "producto_distinto" && <p>Se leyó texto pero no corresponde al producto.</p>}
              {result.capa2.status === "no_legible" && <p>No se pudo leer texto en la carga.</p>}
              {result.capa2.status === "no_aplica" && <p>Desactivada.</p>}
            </div>

            <div className={`layer-card ${VERDICT_CLASS[result.capa3_status]}`}>
              <h4>{VERDICT_ICON[result.capa3_status]} Capa 3 · Referencia</h4>
              {result.capa3.similarity !== null ? (
                <p>Similitud: {Math.round(result.capa3.similarity * 100)}%</p>
              ) : (
                <p>Sin foto de referencia.</p>
              )}
            </div>
          </div>

          <div className="image-box">
            <img src={result.annotated_image} alt="carga anotada" />
            <p className="prov">Verde numerado = contado. Gris fino = descartado por solapamiento.</p>
          </div>

          {result.capa1.suppressed.length > 0 && (
            <details className="suppressed">
              <summary>Ver {result.capa1.suppressed.length} detecciones descartadas y por qué</summary>
              <ul>
                {result.capa1.suppressed.map((s, i) => (
                  <li key={i}>
                    {s.label} — {Math.round(s.confidence * 100)}% — <em>{s.reason}</em>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </section>
      )}
    </div>
  );
}

export default CargoVerifyPanel;
