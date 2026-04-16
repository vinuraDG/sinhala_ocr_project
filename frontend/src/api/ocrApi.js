const API_BASE = "http://localhost:5000";

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error("Backend not ready");
  return response.json();
}

export async function recognizeText(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
    // Do NOT set Content-Type manually — the browser sets it with boundary
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Recognition failed");
  }

  return data;
}

export async function recognizeBase64(base64String) {
  const response = await fetch(`${API_BASE}/predict/base64`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_base64: base64String }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Recognition failed");
  }

  return data;
}