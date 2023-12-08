import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T

from rich.console import Console
console = Console()

#############################
# - Semantic Segmentation - #
#############################
from segmentation.semseg.models import *
from segmentation.semseg.datasets import *
from segmentation.semseg.utils.visualize import draw_text

class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette))
        self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        H, W = image.shape[1:]
        # scale the short side of image to target size
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        # resize the image
        image = T.Resize((nH, nW))(image)
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        # resize to original image size
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        # get segmentation map (value being 0 to num_classes)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        # Save segmentation map
        save_dir = Path('./result')
        torch.save(seg_map, f'{save_dir}/segmentation_map.pt')
        
        # convert segmentation map to color map
        seg_image = self.palette[seg_map].squeeze()
        if overlay: 
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        image = draw_text(seg_image, seg_map, self.labels)
        return image

    @torch.inference_mode()
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
        
    def predict(self, img_fname: str, overlay: bool) -> Tensor:
        image = io.read_image(img_fname)
        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        seg_map = self.postprocess(image, seg_map, overlay)
        return seg_map
    
####################
# - Colorization - #
####################
import random
from glob import glob
import numpy as np
from PIL import Image, ImageFilter, ImageChops, ImageOps, ImageDraw
import onnxruntime
from colorization.transforms import Lambda, Compose, ToTensor, Resize, Normalize
import torch
from torch.nn.functional import interpolate
from PyQt5 import QtGui, QtCore

class InferenceHandler:
    """ TorchServe Handler for PaintsTorch"""

    def __init__(self, model_path="./colorization/SketchColorizationModel.onnx"):

        self.__model = onnxruntime.InferenceSession(model_path)

        self.line_transform = Compose([
            Resize((512, 512)),
            ToTensor(),
            Normalize([0.5], [0.5]),
            Lambda(lambda img: np.expand_dims(img, 0))
        ])
        self.hint_transform = Compose([
            # input must RGBA !
            Resize((128, 128), Image.NEAREST),
            Lambda(lambda img: img.convert(mode='RGB')),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            Lambda(lambda img: np.expand_dims(img, 0))
        ])
        self.line_draft_transform = Compose([
            Resize((128, 128)),
            ToTensor(),
            Normalize([0.5], [0.5]),
            Lambda(lambda img: np.expand_dims(img, 0))
        ])
        self.alpha_transform = Compose([
            Lambda(lambda img: self.get_alpha(img)),
        ])

    def convert_to_pil_image(self, image):
        image = np.transpose(image, (0, 2, 3, 1))
        image = image * 0.5 + 0.5
        image = image * 255
        image = image.astype(np.uint8)[0]
        image = Image.fromarray(image).convert('RGB')
        return image

    def convert_to_pil_line(self, image, size=512):
        image = np.transpose(image, (0, 2, 3, 1))
        image = image * 0.5 + 0.5
        image = image * 255
        image = image.astype(np.uint8)[0]
        image = np.reshape(image, (size, size))
        image = Image.fromarray(image).convert('RGB')
        return image

    def get_alpha(self, hint: Image.Image):
        """
        :param hint:
        :return:
        """
        hint = hint.resize((128, 128), Image.NEAREST)
        hint = np.array(hint)
        alpha = hint[:, :, -1]
        alpha = np.expand_dims(alpha, 0)
        alpha = np.expand_dims(alpha, 0).astype(np.float32)
        alpha[alpha > 0] = 1.0
        alpha[alpha > 0] = 1.0
        alpha[alpha < 1.0] = 0
        return alpha

    def prepare(self, line: Image.Image, hint: Image.Image):
        """
        :param req:
        :return:
        """

        line = line.convert(mode='L')
        alpha = hint.convert(mode='RGBA')
        hint = hint.convert(mode='RGBA')

        w, h = line.size

        alpha = self.alpha_transform(alpha)
        line_draft = self.line_draft_transform(line)
        line = self.line_transform(line)
        hint = self.hint_transform(hint)
        hint = hint * alpha
        hint = np.concatenate([hint, alpha], 1)
        return line, line_draft, hint, (w, h)

    def inference(self, data, **kwargs):
        """
        PaintsTorch inference
        colorization Line Art Image
        :param data: tuple (line, line_draft, hint, size)
        :return: tuple image, size(w,h)
        """
        line, line_draft, hint = data

        inputs_tag = self.__model.get_inputs()
        inputs = {
            inputs_tag[0].name: line,
            inputs_tag[1].name: line_draft,
            inputs_tag[2].name: hint
        }
        image = self.__model.run(None, inputs)[0]
        return image

    def resize(self, image: Image.Image, size: tuple) -> Image.Image:
        """
        Image resize to 512
        :param image: PIL Image data
        :param size:  w,h tuple
        :return: resized Image
        """
        (width, height) = size

        if width > height:
            rate = width / height
            new_height = 512
            new_width = int(512 * rate)
        else:
            rate = height / width
            new_width = 512
            new_height = int(rate * 512)

        return image.resize((new_width, new_height), Image.BICUBIC)

    def postprocess(self, data) -> Image.Image:
        """
        POST processing from inference image Tensor
        :param data: tuple image, size(w,h)
        :return: processed Image json
        """
        pred, size = data
        pred = self.convert_to_pil_image(pred)
        image = self.resize(pred, size)
        return image