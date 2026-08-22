import { useState } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";

function ContainerPanel() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [model, setModel] = useState("yolov8s-oiv7");
  const [conf, setConf] = useState(0.15);
  const [prompt, setPrompt] = useState("shipping container");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const isGroundingDino = model === "grounding-dino-tiny";

  function handleFileChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }

  async function handleSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", model);
    formData.append("conf", conf);
    formData.append("prompt", prompt);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`Error del servidor: ${res.status}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <h1>Probador de modelos de detección de contenedores</h1>

      <div className="controls">
        <label>
          Modelo:
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            <option value="yolov8n-oiv7">yolov8n-oiv7 (rápido, clases fijas)</option>
            <option value="yolov8s-oiv7">yolov8s-oiv7 (más preciso, clases fijas)</option>
            <option value="grounding-dino-tiny">grounding-dino-tiny (texto libre, más lento)</option>
          </select>
        </label>

        {isGroundingDino && (
          <label>
            Prompt de texto:
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="shipping container"
            />
          </label>
        )}

        <label>
          Confianza mínima: {conf}
          <input
            type="range"
            min="0.05"
            max="0.9"
            step="0.05"
            value={conf}
            onChange={(e) => setConf(parseFloat(e.target.value))}
          />
        </label>

        <input type="file" accept="image/*" onChange={handleFileChange} />

        <button onClick={handleSubmit} disabled={!file || loading}>
          {loading ? "Detectando..." : "Detectar"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="images">
        {preview && !result && (
          <div className="image-box">
            <h3>Original</h3>
            <img src={preview} alt="original" />
          </div>
        )}

        {result && (
          <div className="image-box">
            <h3>Resultado ({result.model})</h3>
            <img src={result.annotated_image} alt="resultado" />
          </div>
        )}
      </div>

      {result && (
        <div className="detections">
          <h3>Detecciones ({result.detections.length})</h3>
          {result.detections.length === 0 && <p>No se detectó nada con este umbral de confianza.</p>}
          <ul>
            {result.detections.map((d, i) => (
              <li key={i}>
                <strong>{d.label}</strong> — confianza {(d.confidence * 100).toFixed(1)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default ContainerPanel;
