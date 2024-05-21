import json

import requests
from zerollama.microservices.entrypoints.openai_compatible.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorResponse,
    LogProbs,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
)


class Models(object):
    def __init__(self, client):
        self._client = client
        self.base_url = self._client.base_url

    def list(self):
        r = requests.get(self.base_url + "/v1/models")
        return r.json()

    def retrieve(self, model):
        r = requests.get(self.base_url + f"/v1/models/{model}")
        return r.json()


class Completions(object):
    def __init__(self, client):
        self._client = client
        self.base_url = self._client.base_url

    def create(self, model, messages, stream=False):
        if stream:
            def generate():
                r = requests.post(self.base_url + "/v1/chat/completions",
                                  json={
                                      "model": model,
                                      "messages": messages,
                                      "stream": True
                                  },
                                  stream=True)
                for chunk in r.iter_lines():
                    ccs = ChatCompletionStreamResponse(**json.loads(chunk))
                    if ccs.choices[0].finish_reason is None:
                        yield ccs
            return generate()
        else:
            r = requests.post(self.base_url + "/v1/chat/completions",
                              json={
                                  "model": model,
                                  "messages": messages
                              })
            return ChatCompletionResponse(**r.json())


class Chat(object):
    def __init__(self, client):
        self._client = client
        self.base_url = self._client.base_url
        self.completions = Completions(self._client)


class Embeddings(object):
    def __init__(self, client):
        self._client = client
        self.base_url = self._client.base_url

    def create(self, model, input, encoding_format):
        r = requests.post(self.base_url + "/v1/embeddings",
                          json={
                              "model": model,
                              "input": input
                          })
        return r.json()


class OpenAI(object):
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url

        self.models = Models(self)
        self.chat = Chat(self)
        self.embeddings = Embeddings(self)