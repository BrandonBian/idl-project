import random
from glob import glob
import numpy as np
from PIL import Image, ImageFilter, ImageChops, ImageOps, ImageDraw
import onnxruntime
from transforms import Lambda, Compose, ToTensor, Resize, Normalize
import torch
from torch.nn.functional import interpolate
from PyQt5 import QtGui, QtCore

class InferenceHandler:
    """ TorchServe Handler for PaintsTorch"""

    def __init__(self, model_path="./SketchColorizationModel.onnx"):

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


inference_handler = InferenceHandler()


def predict(line: Image.Image, hint: Image.Image):
    line, line_draft, hint, size = inference_handler.prepare(line, hint)
    pred = inference_handler.inference((line, line_draft, hint))
    image = inference_handler.postprocess((pred, size))
    return image


if __name__ == "__main__":
    # Open image file
    file_path = "../input.png"

    img = Image.open(file_path).convert('RGB')

    if 'png' in file_path.lower():
        img = img.convert('LA')

    line = img.copy()

    # Resize
    width = float(img.size[0])
    height = float(img.size[1])

    if width > height:
        rate = width / height
        new_height = 512
        new_width = int(512 * rate)
    else:
        rate = height / width
        new_width = 512
        new_height = int(rate * 512)

    print(f"Resize: ({width}, {height}) -> ({new_width}, {new_height})")
    img = img.resize((new_width, new_height), Image.BICUBIC)

    # Define color point
    x = int(input("Coordinate x: "))
    y = int(input("Coordinate y: "))
    # TODO: define color
    color = (255, 0, 0)
    width = 4

    # Construct a hint_map of translucent canvas, add the color points
    hint_map = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hint_map)
    draw.line([(x, y), (x + width, y)], fill=color, width=width)
    hint_map.show()
    
    # Colorization with model
    img = predict(img, hint_map)
    img.show()
