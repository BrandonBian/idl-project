# idl-project
Group Project for IDL

## Environment Set-up (tested on Windows 10)
```bash
conda create -n idl-project python=3.6 -y  # Python 3.6 to be compatible with the colorization model
conda activate idl-project

pip install torch  # For now let's just use CPU version of Pytorch since no available version for Python 3.6
pip install pyyaml
pip install torchvision
pip install tabulate
pip install einops
pip install scipy
pip install tqdm
pip install matplotlib
pip install rich
```

## Module 1: Semantic Segmentation
- Download `segformer.b2.ade.pth` from [this link](https://drive.google.com/u/0/uc?id=1AcgEK5aWMJzpe8tsfauqhragR0nBHyPh&export=download) and place it in `semantic-segmentation/`
- For inference, run `python .\inference.py` in `semantic-segmentation/`

- TODO:
  - [x] Save the region coordinates and mapping of each segment, along with its label
  - Support training of model on sketched version of dataset
  - Add utility codes for transforming training data to sketches

## Module 2: NLP

## Module 3: Colorization
- TODO:
  - Extract the inference from the `app/` API
  - Allow command line / argument / config input of color pixel coordinates and RGB values
