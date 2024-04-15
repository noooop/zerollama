
import json
import datetime
from flask import Flask, request, jsonify, Response
from applications.chat_server.client import Client


client = Client()
app = Flask(__name__)

PORT = 9527


def get_timestamp():
    return datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')


@app.route("/")
def hello_world():
    return "zerollama is running"


@app.route("/api/tags", methods=['GET'])
def tags():
    msg = {
        "models": [
            {
                "name": "Qwen/Qwen1.5-0.5B-Chat",
                "model": "Qwen/Qwen1.5-0.5B-Chat"
            }
        ]
    }
    return jsonify(msg)


@app.route("/api/chat", methods=['POST'])
def chat():
    msg = request.get_json(force=True)

    if "model" not in msg:
        result = {"error": "model is required"}
        return jsonify(result)

    model = msg["model"]
    messages = msg.get("messages", list())
    options = msg.get("options", dict())
    stream = msg.get("stream", True)

    data = json.dumps({
        "model": model,
        "messages": messages,
        "options": options,
        "stream": stream
    }).encode('utf8')

    if stream:
        def generate():
            for msg in client.stream_query(port=PORT, data=data):
                content = msg["content"]
                response = json.dumps({"model": model,
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
        msg = client.query(port=PORT, data=data)
        content = msg["content"]
        response = json.dumps({"model": model,
                               "created_at": get_timestamp(),
                               "message": {"role": "assistant", "content": content},
                               "done": True
                               }).encode('utf8')
        return jsonify(response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=11434, debug=True)