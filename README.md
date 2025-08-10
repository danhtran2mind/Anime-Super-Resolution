# Anime Super Resolution 🖼️

> https://github.com/ai-forever/Real-ESRGAN https://github.com/danhtran2mind/Real-ESRGAN-inference
https://github.com/xinntao/Real-ESRGAN https://github.com/danhtran2mind/Real-ESRGAN 
## Introduction

## Key Features

## Dataset
### Source
### Data Structure

```markdown
data/ 📁
├── anime-images-raw/ 📁
│   ├── frame_0001.jpg 📸
│   ├── frame_0001_1.jpg 📷
│   └── ... 📸
├── anime-images-multiscale/ 📁
│   ├── frame_0001T0.png 📸
│   ├── frame_0001T1.png 📸
│   ├── frame_0001T2.png 📸
│   ├── frame_0001T3.png 📸
│   ├── frame_0001_10T0.png 📸
│   ├── frame_0001_10T1.png 📸
│   ├── frame_0001_10T2.png 📸
│   ├── frame_0001_10T3.png 📸
│   └── ... 📸
└── meta_info/ 📁
    └── meta_info_multiscale.txt 📄
```
To continue, see at [Real-ESRGAN Data Processing](#real-esrgan-data-processing-for-training)
## Base Model
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth

## Demonstration

### Interactive Demo

Explore the interactive demo hosted on HuggingFace:
[![HuggingFace Space Demo](https://img.shields.io/badge/HuggingFace-danhtran2mind%2FAnime--Super--Resolution-yellow?style=flat&logo=huggingface)](https://huggingface.co/spaces/danhtran2mind/Anime-Super-Resolution)

Below is a screenshot of the SlimFace Demo GUI:

<img src="./assets/gradio_app_demo.jpg" alt="SlimFace Demo" height="600">

### Run Locally

To run the Gradio application locally at the default address `localhost:7860`, execute:

```bash
python apps/gradio_app.py
```

## Installation
### Clone GitHub Repository
```bash
git clone https://github.com/danhtran2mind/Anime-Super-Resolution
cd Anime-Super-Resolution
```
### Install Dependencies (Training + Inference)
```bash
pip install -e .
```
### Install Dependencies for Inference only
```bash
pip install -r requirements/requirements_inference.txt
```
### Execute Sripts

#### Download Model checkpoints

- Download only model checkpoint for Inference
    ```bash
    python scripts/download_ckpts.py
    ```

- For training from `Real-ESRGAN` Model checkpoint

    ```bash
    python scripts/download_ckpts.py --base_model_only
    ```
- For training from `Real-ESRGAN-Anime-finetuning` Model checkpoint
    ```bash
    python scripts/download_ckpts.py --full_ckpts
    ```
More detail you can read at [Download Model Checkpoints](docs/scripts/download_model_ckpts.md).

#### Setup Third Party
```bash
    python scripts/setup_third_party.py
```

#### Download Dataset
```bash
    python scripts/download_datasets.py \
        --dataset_id "<huggingface_dataset_id>"
        --huggingface_token "<your_huggingface_token>"
```
More detail you can read at [Download Dataset](docs/scripts/download_dataset_doc.md).

## Usage

### Real-ESRGAN Data Processing (for Training)
First you need see the data structure at [Data Structure](#data-structure). Then execute below scripts to process dataset.
- Create `multiscale` Folder
    ```bash
    python src/third_party/Real-ESRGAN/scripts/generate_multiscale_DF2K.py \
        --input ./data/anime-images-raw \
        --output ./data/anime-images-multiscale
    ```
- Create `meta_info_multiscale.txt`
    ```bash
    python src/third_party/Real-ESRGAN/scripts/generate_meta_info.py \
    --input ./data/anime-images-raw ./data/anime-images-multiscale \
    --root ./data ./data \
    --meta_info "./data/meta_info/meta_info_multiscale.txt"
    ```

### Training
#### Training Script
```bash
python src/anime_super_resolution/train.py \
    --config "<your_model_config_yml_path>" \
    --auto_resume
```

#### Additional Arguments
For more details and available options, refer to the [Training Document](docs/training/training_doc.md).


You can see [Real-ESRGAN Training](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/Training.md), and 
[BasicSR Training Options](https://github.com/danhtran2mind/BasicSR/blob/master/basicsr/utils/options.py) for more details.

### Inference
#### Inference Script
<!-- ```bash
python src/anime_super_resolution/infer.py
``` -->
```bash
python src/anime_super_resolution/infer.py \
    --input_path tests/test_data/input_image.png \
    --output_dir tests/test_data \
    --suffix real_esrgan_anime \
    --outscale 2 \
    --model_path ckpts/Real-ESRGAN-Anime-finetuning/net_g_latest.pth
```

#### Additional Arguments
For more details and available options, refer to the [Inference Document](docs/inference/inference_doc.md).

## Environment

SlimFace requires the following environment:

- **Python**: 3.10 or higher
- **Key Libraries**: Refer to [Requirements Compatible](./requirements/requirements_compatible.txt) for compatible dependencies.

## Project Credits and Resources

- This project leverages code from:

    > The Original Real-ESRGAN by [![GitHub](https://img.shields.io/badge/GitHub-xinntao-blue?style=flat&logo=github)](https://github.com/xinntao) at [![Built on Real-ESRGAN](https://img.shields.io/badge/Built%20on-xinntao%2FReal--ESRGAN-blue?style=flat&logo=github)](https://github.com/xinntao/Real-ESRGAN). Our own bug fixes and enhancements are available at [![Real-ESRGAN Enhancements](https://img.shields.io/badge/GitHub-danhtran2mind%2FReal--ESRGAN-blue?style=flat&logo=github)](https://github.com/danhtran2mind/Real-ESRGAN).

    > The Inference code by 
    [![GitHub](https://img.shields.io/badge/GitHub-ai--forever-blue?style=flat&logo=github)](https://github.com/ai-forever) at [![Built on Real-ESRGAN](https://img.shields.io/badge/Built%20on-ai--forever%2FReal--ESRGAN-blue?style=flat&logo=github)](https://github.com/ai-forever/Real-ESRGAN).
    Our own bug fixes and enhancements are available at [![Real-ESRGAN Enhancements](https://img.shields.io/badge/GitHub-danhtran2mind%2FReal--ESRGAN--inference-blue?style=flat&logo=github)](https://github.com/danhtran2mind/Real-ESRGAN-inference)

- You can explore more Model Hubs at:

    > HuggingFace Model Hub: [![ai-forever Real-ESRGAN Model](https://img.shields.io/badge/HuggingFace-ai--forever%2FReal--ESRGAN-yellow?style=flat&logo=hugggingface)](https://huggingface.co/ai-forever/Real-ESRGAN). Real-ESRGAN Model releases: [![Real-ESRGAN releases](https://img.shields.io/badge/GitHub-Real--ESRGAN%2Freleases-blue?style=flat&logo=github)](https://github.com/xinntao/Real-ESRGAN/releases)

<!-- https://github.com/ai-forever/Real-ESRGAN https://github.com/danhtran2mind/Real-ESRGAN-inference
https://huggingface.co/ai-forever/Real-ESRGAN
https://github.com/xinntao/Real-ESRGAN https://github.com/danhtran2mind/Real-ESRGAN 
https://github.com/xinntao/Real-ESRGAN/releases -->