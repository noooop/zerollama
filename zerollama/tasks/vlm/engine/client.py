
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.tasks.vlm.protocol import PROTOCOL
from zerollama.tasks.vlm.protocol import VLMChatCompletionRequest, VLMChatCompletionResponse

CLIENT_VALIDATION = True


class VLMClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def chat(self, name, messages, images, options=None):
        method = "inference"
        data = {"model": name,
                "messages": messages,
                "images": images,
                "options": options or dict(),
                "stream": False}
        if CLIENT_VALIDATION:
            data = VLMChatCompletionRequest(**data).dict()

        rep = self.query(name, method, data)
        if rep is None:
            raise RuntimeError(f"VLM [{name}] server not found.")

        if rep.state != "ok":
            raise RuntimeError(f"VLM [{name}] error, with error msg [{rep.msg}]")

        return VLMChatCompletionResponse(**rep.msg)


if __name__ == '__main__':
    import PIL.Image
    import numpy as np

    import os
    from pathlib import Path

    vlm_test_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent / "static/test_sample/vlm"

    model_name = "openbmb/MiniCPM-V"

    client = VLMClient()
    print("=" * 80)
    print(f"Wait {model_name} available")
    client.wait_service_available(model_name)
    print(client.get_services(model_name))

    print("=" * 80)
    print('ZeroVLMInferenceEngine support_methods')
    print(client.support_methods(model_name))
    print(client.info(model_name))

    messages = [
        {
            "role": "user",
            "content": "这个图片的内容是什么。"
        }
    ]

    images = [vlm_test_path / "monday.jpg"]

    images = [PIL.Image.open(path).convert("RGB") for path in images]
    images = [np.array(image) for image in images]

    print("="*80)
    print("stream == False")
    response = client.chat(model_name, messages, images)
    print(response.content)