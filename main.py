import yaml
import json
import numpy as np
import torch
import torch.nn.functional as F

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

    print("\n----------------------------------------")
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

    with console.status("[bright_green]Processing..."):
        for sampling_points in [5, 10, 15, 20, 25, 30]:
            for stroke_size in [2, 4, 6, 8]:
                print(f"Generating candidate with configuration: sampling points = {sampling_points}, stroke size = {stroke_size}")

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

                # Create a hint map of translucent canvas, and add color points based on user requirements
                hint_map = Image.new("RGBA", (img.size), (0, 0, 0, 0))
                draw = ImageDraw.Draw(hint_map)

                # Place color points on hint map
                unique_labels = color_mapping.keys()
                selected_coordinates = {}

                max_area = max(len(np.column_stack(np.where(segmentation_map == label))) for label in unique_labels)

                for label in unique_labels:
                    # Get coordinates of all pixels with current label
                    coords = np.column_stack(np.where(segmentation_map == label))

                    # Scale the number of sampled color points based on the area
                    # NOTE: larger areas will have more color points, minimum points hardcoded as 3
                    scaled_points = max(3, int(sampling_points * (len(coords) / max_area)))

                    # Scale the coloring stroke
                    # NOTE: smaller areas will have larger stroke size, minimum stroke hardcoded as 3
                    # NOTE: not used for now
                    scaled_stroke = max(3, int(stroke_size * (1 - (len(coords) / max_area))))

                    # Randomly select [scaled_points] coordinates
                    if coords.shape[0] > scaled_points:
                        indices = np.random.choice(coords.shape[0], scaled_points, replace=False)
                    else:
                        indices = np.random.choice(coords.shape[0], scaled_points, replace=True)  # Allow repeats if less than 5

                    selected_coordinates[label] = coords[indices]
                
                for label, coordinates in selected_coordinates.items():
                    for x, y in coordinates:
                        # Fetch RGB from color mapping
                        rgb = color_mapping[int(segmentation_map[x][y])][1]
                        # Color the coordinate on hint map
                        draw.line([(x, y), (x + stroke_size, y)], fill=rgb, width=stroke_size)
                
                # Colorization with model
                assert img.size == hint_map.size, f"(Img Size = {img.size}) != (Hint Map Size = {hint_map.size})"
                img = predict(img, hint_map, inference_handler)
                
                # Save results
                save_dir = Path("./result/candidates")
                save_dir.mkdir(exist_ok=True)

                hint_map.save(f"{save_dir}/hintmap_sampling{sampling_points}_stroke{stroke_size}.png")
                img.save(f"{save_dir}/result_sampling{sampling_points}_stroke{stroke_size}.png")

    print("\n--------------")
    print("--- Done! ---")
    print("-------------")