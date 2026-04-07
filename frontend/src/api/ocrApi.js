import axios from "axios";

const BASE_URL = "http://localhost:5000";

export async function recognizeText(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await axios.post(`${BASE_URL}/predict`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 30000,
  });

  return response.data;
}

export async function healthCheck() {
  const response = await axios.get(`${BASE_URL}/health`, { timeout: 5000 });
  return response.data;
}
