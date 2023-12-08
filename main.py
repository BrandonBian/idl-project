import yaml
import json
from pathlib import Path
from utils import SemSeg, console, InferenceHandler

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

    # TODO
    inference_handler = InferenceHandler()