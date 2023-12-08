import yaml
import json
from pathlib import Path
from utils import SemSeg, console

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