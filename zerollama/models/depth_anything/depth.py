from zerollama.tasks.mono_estimation.depth.interface import DepthEstimationModel


class DepthAnythingV2(DepthEstimationModel):
    family = "Depth-Anything-V2"
    header = ["name", "use_hf_only"]
    info = [
        ["depth-anything/Depth-Anything-V2-Small-hf",         True],
        ["depth-anything/Depth-Anything-V2-Base-hf",          True],
        ["depth-anything/Depth-Anything-V2-Large-hf",         True],
    ]
    inference_backend = "zerollama.models.depth_anything.backend.depth:DepthAnything"


if __name__ == '__main__':
    import requests
    import numpy as np
    from PIL import Image

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    for model_name, *_ in DepthAnythingV2.info:
        model = DepthAnythingV2.get_model(model_name, local_files_only=False)
        model.load()

        output = model.estimation(image).depth

        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)

        depth.save(f'result-{model_name.replace("/", "-")}.jpg')