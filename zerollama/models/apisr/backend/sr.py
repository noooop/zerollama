
import os
import cv2
import torch
from pathlib import Path
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from zerollama.tasks.super_resolution.protocol import SRResponse
from zerollama.tasks.super_resolution.interface import SuperResolutionInterface
from zerollama.tasks.super_resolution.collection import get_model_config_by_name, get_model_by_name

apisr_path = Path(os.path.dirname(__file__)).parent.parent.parent.parent.parent / "APISR"


class APISR(SuperResolutionInterface):
    def __init__(self, model_name, float16_inference=False):
        model_config = get_model_config_by_name(model_name)

        if model_config is None:
            raise FileNotFoundError(f"model [{model_name}] not supported.")

        self.model_name = model_name
        self.model_config = model_config
        self.model_info = self.model_config.info

        self.scale = self.model_info["scale"]
        self.weight_path = apisr_path / "pretrained" / self.model_info["weight_path"]

        self.float16_inference = float16_inference

        self.model = None
        self.weight_dtype = None
        self.n_concurrent = 1

    def load(self):
        import sys
        sys.path.append(str(apisr_path))

        from test_code.test_utils import load_grl, load_rrdb, load_dat, load_cunet

        if "GRL" in self.model_name:
            self.model = load_grl(self.weight_path, scale=self.scale)  # GRL for Real-World SR only support 4x upscaling

        elif "DAT" in self.model_name:
            self.model = load_dat(self.weight_path, scale=self.scale)  # GRL for Real-World SR only support 4x upscaling

        elif "RRDB" in self.model_name:
            self.model = load_rrdb(self.weight_path, scale=self.scale)  # Can be any size

        # Define the weight type
        if self.float16_inference:
            torch.backends.cudnn.benchmark = True
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32
        self.weight_dtype = weight_dtype

        self.model = self.model.to(dtype=weight_dtype)

    @torch.no_grad()
    def sr(self, img_lr, **options):
        downsample_threshold = options.get("downsample_threshold", -1)
        crop_for_4x = options.get("crop_for_4x", True)

        h, w, c = img_lr.shape

        short_side = min(h, w)
        if downsample_threshold != -1 and short_side > downsample_threshold:
            resize_ratio = short_side / downsample_threshold
            img_lr = cv2.resize(img_lr, (int(w / resize_ratio), int(h / resize_ratio)), interpolation=cv2.INTER_LINEAR)

        if crop_for_4x:
            h, w, _ = img_lr.shape
            if h % 4 != 0:
                img_lr = img_lr[:4 * (h // 4), :, :]
            if w % 4 != 0:
                img_lr = img_lr[:, :4 * (w // 4), :]

        img_lr = ToTensor()(img_lr).unsqueeze(0).cuda()  # Use tensor format
        img_lr = img_lr.to(dtype=self.weight_dtype)

        super_resolved_img = self.model(img_lr)

        with torch.cuda.amp.autocast():
            grid = make_grid(super_resolved_img)
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            return SRResponse(model=self.model_name, image=ndarr)



