

from flask import Flask, request, jsonify
from client import Client

client = Client()
app = Flask(__name__)


@app.route("/")
def hello_world():
    return "zerollama is running"

@app.route("/api/tags", methods = ['GET'])
def tags():
    msg = {
        "models": [
            {
                "name": "qwen:0.5B",
                "model": "qwen:0.5B",
                "modified_at": "2024-04-11T15:04:27.6087797+08:00",
                "size": 2330093361,
                "digest": "",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "qwen2",
                    "families": [
                        "qwen2"
                    ],
                    "parameter_size": "0.5B",
                    "quantization_level": ""
                }
            }
        ]
    }
    return jsonify(msg)


@app.route("/api/chat", methods=['POST'])
def chat():
    msg = request.get_json(force=True)
    result = client.query(port="9527",
                          messages=msg["messages"])
    return jsonify(result)


@app.route("/v1/chat/completions", methods=['POST'])
def openai_chat():
    msg = request.get_json(force=True)
    result = client.query(port="9527",
                          messages=msg["messages"])
    return jsonify(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=11434, debug=True)