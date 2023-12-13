import yaml
import json
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image, ImageDraw
from utils import SemSeg, console, InferenceHandler, predict
import time
import ast
import openai
import textwrap
import argparse

class OpenAIEngine():
  def __init__(self, model_name):
    self.model_name = model_name

  def score(self, text):
    """Tokenizes and scores a piece of text.

    This only works for the OpenAI models which support the legacy `Completion`
    API.

    The score is log-likelihood. A higher score means a token was more
    likely according to the model.

    Returns a list of tokens and a list of scores.
    """
    response = openai.Completion.create(
        engine=self.model_name,
        prompt=text,
        max_tokens=0,
        logprobs=1,
        echo=True)

    tokens = response["choices"][0]["logprobs"]["tokens"]
    logprobs = response["choices"][0]["logprobs"]["token_logprobs"]
    if logprobs and logprobs[0] is None:
      # GPT-3 API does not return logprob of the first token
      logprobs[0] = 0.0
    return tokens, logprobs

  def perplexity(self, text):
    """Compute the perplexity of the provided text."""
    completion = openai.Completion.create(
        model=self.model_name,
        prompt=text,
        logprobs=0,
        max_tokens=0,
        temperature=1.0,
        echo=True)
    token_logprobs = completion['choices'][0]['logprobs']['token_logprobs']
    nll = np.mean([i for i in token_logprobs if i is not None])
    ppl = np.exp(-nll)
    return ppl

  def generate(self,
               prompt,
               top_p=1.0,
               num_tokens=32,
               num_samples=1,
               frequency_penalty=0.0,
              presence_penalty=0.0):
    """Generates text given the provided prompt text.

    This only works for the OpenAI models which support the legacy `Completion`
    API.

    If num_samples is 1, a single generated string is returned.
    If num_samples > 1, a list of num_samples generated strings is returned.
    """
    response = openai.completions.create(
      model=self.model_name,
      prompt=prompt,
      temperature=1.0,
      max_tokens=num_tokens,
      top_p=top_p,
      n=num_samples,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty,
      logprobs=1,
    )
    # outputs = [r["text"] for r in response["choices"]]
    # return outputs[0] if num_samples == 1 else outputs
    return response.choices[0].text


  def chat_generate(self,
                    previous_messages,
                    top_p=1.0,
                    num_tokens=32,
                    num_samples=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0):
    response = openai.ChatCompletion.create(
      model=self.model_name,
      messages=previous_messages,
      temperature=1.0,
      max_tokens=num_tokens,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty,
      n=num_samples,
    )
    return response

def transform_user_input(user_input):
    # user_input = 'I want the door to be green, the painting to be pink, the lamp to be yellow, the sconce to be navy-blue, the sofa to be blue, the coffee table to be green, and the book to be orange'
    prompt = f"The user input is {user_input}. Correctly identify the objects and its corresponding colors and reformat the input into a json string where the key is the object and the value is the color."
    json_str = engine.generate(prompt, num_tokens=512, num_samples=1, top_p=0.6)
    res = json.loads(json_str)
    return res
def transform_color_rgb(color):
    prompt = f"convert color {color} to rgb value. return the rgb value as a tuple only"
    str_rgb = engine.generate(prompt, num_tokens=512, num_samples=1, top_p=0.6)
    rgb_tuple = ast.literal_eval(str_rgb.strip())
    return rgb_tuple

if __name__ == "__main__":
    # OpenAI API
    OPENAI_SECRET_KEY = "input_your_own_key"
    openai.api_key = OPENAI_SECRET_KEY
    MODEL_NAME = "text-davinci-002" #"davinci"
    engine = OpenAIEngine(MODEL_NAME)

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--example', action='store_true', help='Indicate this is an example')
    args = parser.parse_args()
    
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
    
    if not args.example:
        objects = list(label_mapping.values())
        print("We detected the following objects:", objects)
        user_input = input("\n>> Please describe your colorization requirements of the detect objects: ")
        final_res = transform_user_input(user_input)
        res_objects = final_res.keys()
        diff1 = [item for item in objects if item not in res_objects]
        while len(diff1)>0:
            diff_string = ', '.join(diff1)
            user_input = input(f"\n>> Seems like you forget to colorize {diff_string}. Please describe your colorization requirement of these objects: ")
            res = transform_user_input(user_input)
            final_res.update(res)
            res_objects = final_res.keys()
            diff1 = [item for item in objects if item not in res_objects]
        
        color_to_rgb = {
            "red": (255, 0, 0),
            "green": (0, 128, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "pink": (255, 192, 203),
            "brown": (165, 42, 42),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "gray": (128, 128, 128),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "lime": (0, 255, 0),
            "maroon": (128, 0, 0),
            "olive": (128, 128, 0),
            "navy": (0, 0, 128),
            "teal": (0, 128, 128),
            "silver": (192, 192, 192),
            "gold": (255, 215, 0)
        }

        color_mapping = {}
        for idx,obj in label_mapping.items():
            color = final_res[obj]
            if color in color_to_rgb.keys():
                color_rgb = color_to_rgb[color]
            else:
                color_rgb = transform_color_rgb(color)
                color_to_rgb[color] = color_rgb
                time.sleep(5)
            color_mapping[int(idx)] = (obj,color_rgb)
    else:
        print(">> Using example color requirements")
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

                    try:
                        # Randomly select [scaled_points] coordinates
                        if coords.shape[0] > scaled_points:
                            indices = np.random.choice(coords.shape[0], scaled_points, replace=False)
                        else:
                            indices = np.random.choice(coords.shape[0], scaled_points, replace=True)  # Allow repeats if less than 5

                        selected_coordinates[label] = coords[indices]
                    except:
                        pass
                
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
