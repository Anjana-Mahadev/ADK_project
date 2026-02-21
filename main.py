from flask import Flask, request, jsonify
from agent import root_agent

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "service": "GCP RAG Agent"
    })


@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json()

    if not payload or "question" not in payload:
        return jsonify({
            "error": "Request must contain 'question'"
        }), 400

    try:
        response = root_agent.run(payload["question"])

        return jsonify({
            "question": payload["question"],
            "answer": response
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
