# idl-project
Group Project for IDL

## Environment Set-up (tested on Windows 10)
- python >= 3.7
- torch >= 1.8.1
- torchvision >= 0.9.1
```bash
# Create environment with Python 3.7 to be compatible with the colorization model
conda create -n idl-project python=3.7 -y
conda activate idl-project

# Since there is no suitable pytorch for Python 3.6, 
# let's just use CPU version of Pytorch for inference
pip install torch

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
pip install wandb
pip install onnxruntime
pip install pyqt5
pip install openai==1.3.7
```

## Usage
- **Step 1**: set up and activate the conda environment following instructions above
- **Step 2**: create `segmentation/models/` directory, enter it, and run `gdown https://drive.google.com/u/1/uc?id=1AcgEK5aWMJzpe8tsfauqhragR0nBHyPh` to get the pre-trained model on full ADE20K dataset
- **Step 3**: enter `colorization`, and run `gdown https://drive.google.com/u/2/uc?id=1G7nN4PL48TB6aJjCn7F6tJSM_d7pZ3_4` to get a pre-trained model
- **Step 4**: run `python main.py [--example, for using example color mapping for the existing input.png]` and follow the instructions. Results will be saved in `result/`

## Module 1: Semantic Segmentation
### Preparation - Training Data
- **Full MIT Scene Parsing ADE 20K Benchmark**: http://sceneparsing.csail.mit.edu/
- **Our adapted/filtered room-specific dataset**: [Google Drive](https://drive.google.com/file/d/1-W-A9gDkVitq7lcGg2srm01DaB5nPsL8/view?usp=drive_link)
  - Download: `gdown https://drive.google.com/u/2/uc?id=1-W-A9gDkVitq7lcGg2srm01DaB5nPsL8`
  - Unzip and place into `semantic-segmentation/data/`
- **Our filtered room-specific dataset, but converted to sketches**: [Google Drive](https://drive.google.com/file/d/1p3D5Y6X89SIOhO7ostJM6R6G1xKPLPxI/view?usp=drive_link)
  - Download: `gdown https://drive.google.com/u/2/uc?id=1p3D5Y6X89SIOhO7ostJM6R6G1xKPLPxI`
  - Unzip and place into `semantic-segmentation/data/`

### Preparation - Models
- **Backbone model** (Mit-B2) for training and finetuning: [Google Drive](https://drive.google.com/file/d/1Ju_8VWh8aG7mKrvfshRCfwNkk1qYIEvl/view?usp=drive_link)
  - Download: `gdown https://drive.google.com/u/2/uc?id=1Ju_8VWh8aG7mKrvfshRCfwNkk1qYIEvl`
  - Place into `semantic-segmentation/models/`
- **Pretrained model - original** which is provided by the authors and trained on the full original ADE-20K dataset: [Google Drive](https://drive.google.com/u/0/uc?id=1AcgEK5aWMJzpe8tsfauqhragR0nBHyPh&export=download)
  - Download: `gdown https://drive.google.com/u/1/uc?id=1AcgEK5aWMJzpe8tsfauqhragR0nBHyPh`
  - Place into `semantic-segmentation/models/`
- **Pretrained model - rooms only** which is trained using Mit-B2 backbone on our filtered room-specific dataset
  - **SegRooms_100Epochs_32mIoU**: 
    - [Google Drive](https://drive.google.com/file/d/1IQNBteYCB1R7kYUlO-uNDOFh2PR80KCP/view?usp=drive_link)
    - `gdown https://drive.google.com/u/2/uc?id=1IQNBteYCB1R7kYUlO-uNDOFh2PR80KCP`
  - **SegRooms_250Epochs_29mIoU**:
    - [Google Drive](https://drive.google.com/file/d/1rrt5m3GeWLyvVhkQ2ibXzqQhs5kOQAJT/view?usp=drive_link)
    - `gdown https://drive.google.com/u/2/uc?id=1rrt5m3GeWLyvVhkQ2ibXzqQhs5kOQAJT`
- **Pretrained model - room sketches only**

### Usage
- For **Training**, run `python ./train.py --name [experiment_name]`
  - Use your own wandb API key in `train.py -> wandb.login()`
  - Make sure you have CUDA GPU available for Pytorch, otherwise it is extremely slow on CPU
  - Recommended to use AWS EC2 for training
  - Validation is performed after per training epoch, and the best model weights is saved
- For **inference**, run `python ./inference.py` in `semantic-segmentation/`

## Module 2: Colorization
- Preparation - Models
  - Pretrained model by the authors on **anime-specific** images: Download `SketchColorizationModel.onnx` from [Github Release](https://github.com/rapidrabbit76/SketchColorization/releases), or use the following command to download from our Google Drive
    - Download: `gdown https://drive.google.com/u/2/uc?id=1G7nN4PL48TB6aJjCn7F6tJSM_d7pZ3_4`
    - Place into `colorization/`
