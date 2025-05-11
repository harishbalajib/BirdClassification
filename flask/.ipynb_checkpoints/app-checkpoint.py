from flask import Flask, jsonify, request
import requests
import os

app = Flask(__name__)

# Get FastAPI URL from environment or default
FASTAPI_URL = os.getenv("FASTAPI_SERVER_URL", "http://fastapi_server:8000")

@app.route("/")
def home():
    return "Flask is running. Use /ping or /test-predict."

@app.route("/ping")
def ping_fastapi():
    try:
        response = requests.get(f"{FASTAPI_URL}/docs")
        return jsonify({
            "fastapi_url": FASTAPI_URL,
            "status": "reachable",
            "code": response.status_code
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test-predict", methods=["GET"])
def test_predict():
    audio_path = "test.wav"
    if not os.path.exists(audio_path):
        return jsonify({"error": "test.wav not found in container"}), 400

    try:
        with open(audio_path, "rb") as f:
            files = {'file': (audio_path, f, 'audio/wav')}
            response = requests.post(f"{FASTAPI_URL}/predict", files=files)
            return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
