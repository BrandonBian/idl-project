import yaml
import json
import torch
from pathlib import Path
from PIL import Image, ImageDraw
from utils import SemSeg, console, InferenceHandler, predict

if __name__ == "__main__":
    print("-------------------------------------------")
    print("--- Step 1: Input your sketch directory ---")
    print("-------------------------------------------")

    with open("./config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    image_dir = input("Sketch directory: ")
    cfg['TEST']['FILE'] = image_dir

    save_dir = Path("./result")
    save_dir.mkdir(exist_ok=True)

    print("\n-------------------------------------")
    print("--- Step 2: Semantic segmentation ---")
    print("-------------------------------------")

    semseg = SemSeg(cfg)

    with console.status("[bright_green]Processing..."):
        segmap = semseg.predict(image_dir, cfg['TEST']['OVERLAY'])
        segmap.save(save_dir / f"semantic_segmentation.png")

    print("\n--------------------------------------")
    print("--- Step 3: User requirement parsing ---")
    print("----------------------------------------")

    with open("./result/label_mapping.json", 'r') as file:
        label_mapping = json.load(file)

    # TODO: add NLP
    objects = list(label_mapping.values())
    print("We detected the following objects:", objects)

    # TODO: NLP's final result should be something like the following
    color_mapping = {
        0: ("wall", (200, 200, 200)),     # Gray wall
        3: ("floor", (150, 100, 50)),     # Brown wooden floor
        5: ("ceiling", (255, 255, 255)),  # White ceiling
        8: ("windowpane", (135, 206, 235)), # Light Blue windowpane
        14: ("door", (160, 82, 45)),      # Brown wooden door
        15: ("table", (139, 69, 19)),     # Dark Brown table
        18: ("curtain", (220, 20, 60)),   # Red curtain
        22: ("painting", (233, 150, 122)), # Light Red painting
        23: ("sofa", (178, 34, 34)),      # Dark Red sofa
        24: ("shelf", (210, 105, 30)),    # Brown shelf
        35: ("wardrobe", (128, 0, 0)),    # Maroon wardrobe
        36: ("lamp", (255, 215, 0)),      # Gold lamp
        39: ("cushion", (255, 160, 122)), # Light Coral cushion
        64: ("coffee table", (139, 69, 19)), # Brown coffee table
        67: ("book", (0, 0, 255)),        # Blue book
        69: ("bench", (85, 107, 47)),     # Dark Olive Green bench
        132: ("sculpture", (192, 192, 192)), # Silver sculpture
        134: ("sconce", (218, 165, 32)),  # Golden Rod sconce
        135: ("vase", (123, 104, 238))    # Medium Slate Blue vase
    }

    print("\n----------------------------")
    print("--- Step 4: Colorization ---")
    print("----------------------------")

    # Load input image
    inference_handler = InferenceHandler()

    img = Image.open(image_dir)
    if 'png' in image_dir.lower():
        img = img.convert('LA')
    
    line = img.copy()

    # Load pixel-level semantic segmentation predictions
    segmentation_map = torch.load('./result/segmentation_map.pt')
    segmentation_map = torch.squeeze(segmentation_map).transpose(0, 1)
    assert segmentation_map.shape == img.size

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
    img = predict(img, hint_map, inference_handler)
    img.show()