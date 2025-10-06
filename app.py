from pathlib import Path
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE = Path(__file__).resolve().parent
PIPE_PATH = BASE / "models" / "pipeline.pkl"
pipe = joblib.load(PIPE_PATH)  # one fitted artifact

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or data.get("message") or "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        p = float(pipe.predict_proba([text])[0][1])  # phishing probability
        label = "Phishing" if p >= 0.5 else "Likely Safe"
        return jsonify({"label": label, "score": round(p, 3)})
    except Exception as e:
        app.logger.exception("Prediction error")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

@app.route("/health")
def health():
    return jsonify({"ok": True})

@app.route("/debug")
def debug():
    tfidf = pipe.named_steps.get("tfidf")
    return jsonify({
        "ok": True,
        "has_idf": hasattr(tfidf, "idf_"),
        "vocab_size": int(len(getattr(tfidf, "vocabulary_", {}))),
    })

if __name__ == "__main__":
    app.run(debug=True)
