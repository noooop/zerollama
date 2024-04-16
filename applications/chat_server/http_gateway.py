
import json
import datetime
from flask import Flask, request, jsonify, Response
from zerollama.core.framework.inference_engine.client import ChatClient


chat_client = ChatClient()
app = Flask(__name__)


def get_timestamp():
    return datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')


@app.route("/")
def hello_world():
    return "zerollama is running"


@app.route("/api/tags", methods=['GET'])
def tags():
    response = chat_client.get_service_names()
    services = response["msg"]["service_names"]

    msg = {
        "models": [
            {"name": s, "model": s}
            for s in services
        ]
    }
    return jsonify(msg)


@app.route("/api/chat", methods=['POST'])
def chat():
    msg = request.get_json(force=True)

    if "model" not in msg:
        result = {"error": "model is required"}
        return jsonify(result)

    name = msg["model"]
    messages = msg.get("messages", list())
    options = msg.get("options", dict())
    stream = msg.get("stream", True)

    if stream:
        def generate():
            for msg in chat_client.stream_chat(name, messages, options):
                content = msg["content"]
                response = json.dumps({"model": name,
                                       "created_at": get_timestamp(),
                                       "message": {"role": "assistant", "content": content},
                                       "done": False
                                       })
                yield response
                yield "\n"

            response = json.dumps({
                "model": "qwen:0.5b-chat",
                "created_at": get_timestamp(),
                "message": {"role": "assistant", "content": ""},
                "done": True
            })
            yield response

        return Response(generate(), mimetype="application/x-ndjson")
    else:
        msg = chat_client.chat(name, messages, options)
        content = msg["content"]
        response = json.dumps({"model": name,
                               "created_at": get_timestamp(),
                               "message": {"role": "assistant", "content": content},
                               "done": True
                               }).encode('utf8')
        return jsonify(response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=11434, debug=True)