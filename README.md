# idl-project
Group Project for IDL

## Environment Set-up (tested on Windows 10)
- python >= 3.6
- torch >= 1.8.1
- torchvision >= 0.9.1
```bash
# Create environment with Python 3.6 to be compatible with the colorization model
conda create -n idl-project python=3.6 -y
conda activate idl-project

# Find your version from: https://pytorch.org/get-started/previous-versions/
# NOTE: my PC has CUDA == 11.3, check yours using "nvcc --version"
# NOTE: using Pytorch 1.9 (pretty old version) for compatibility with Python 3.6
# NOTE: after installing Pytorch, make sure "torch.cuda.is_available() == True"
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install pyyaml
pip install torchvision
pip install tabulate
pip install einops
pip install scipy
pip install tqdm
pip install matplotlib
pip install rich
pip install tensorboard
pip install chardet
pip install gdown
```

## Module 1: Semantic Segmentation
### Preparation - Training Data
- Full MIT Scene Parsing ADE 20K Benchmark (not used): http://sceneparsing.csail.mit.edu/
- Our adapted/filtered room-specific dataset (used instead): [Google Drive](https://drive.google.com/file/d/1-W-A9gDkVitq7lcGg2srm01DaB5nPsL8/view?usp=drive_link)
  - Download using `gdown`: `gdown https://drive.google.com/u/2/uc?id=1-W-A9gDkVitq7lcGg2srm01DaB5nPsL8`
  - Unzip and place into `semantic-segmentation/data/`

### Preparation - Models
- Download `segformer.b2.ade.pth` from [this link](https://drive.google.com/u/0/uc?id=1AcgEK5aWMJzpe8tsfauqhragR0nBHyPh&export=download) and place it in `semantic-segmentation/`
  - If using terminal, install gdown `pip install gdown`, and run `gdown https://drive.google.com/u/1/uc?id=1AcgEK5aWMJzpe8tsfauqhragR0nBHyPh`

### Usage
- For inference, run `python .\inference.py` in `semantic-segmentation/`


## Module 2: NLP

## Module 3: Colorization
- Download `SketchColorizationModel.onnx` from [this link](https://github.com/rapidrabbit76/SketchColorization/releases) and place it in `colorization\`

- TODO:
  - Extract the inference from the `app/` API
  - Allow command line / argument / config input of color pixel coordinates and RGB values
