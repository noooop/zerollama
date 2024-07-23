
from zerollama.core.framework.nameserver.client import ZeroClient
from zerollama.tasks.text2image.protocol import PROTOCOL
from zerollama.tasks.text2image.protocol import Text2ImageRequest, Text2ImageResponse

CLIENT_VALIDATION = True


class Text2ImageClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def generate(self, name, prompt, negative_prompt=None, options=None):
        method = "inference"
        data = {"model": name,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "options": options or dict()}
        if CLIENT_VALIDATION:
            data = Text2ImageRequest(**data).dict()

        rep = self.query(name, method, data)
        if rep is None:
            raise RuntimeError(f"{self.__class__.__class__} [{name}] server not found.")

        if rep.state != "ok":
            raise RuntimeError(f"{self.__class__.__class__} [{name}] error, with error msg [{rep.msg}]")

        return Text2ImageResponse(**rep.msg)


if __name__ == '__main__':
    import os
    from PIL import Image
    from pathlib import Path

    vlm_test_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent / "static/test_sample/vlm"

    model_name = "Tencent-Hunyuan/HunyuanDiT-Diffusers"

    client = Text2ImageClient()
    print("=" * 80)
    print(f"Wait {model_name} available")
    client.wait_service_available(model_name)
    print(client.get_services(model_name))

    print("=" * 80)
    print('ZeroText2ImageInferenceEngine support_methods')
    print(client.support_methods(model_name))
    print(client.info(model_name))

    prompt = "可爱猫猫图"
    image = client.generate(model_name, prompt, options={"height": 1280, "width": 768, "num_inference_steps": 50}).image
    image = Image.fromarray(image)

    image.save(f'result-{model_name.replace("/", "-")}.jpg')