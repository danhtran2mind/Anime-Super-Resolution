# Anime Super Resolution 🖼️

## Introduction

## Key Features

## Dataset

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
[Real-ESRGAN Data Processing](#real-esrgan-data-processing)
## Base Model


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

```bash
python scripts/download_ckpts.py
```
```bash
python scripts/setup_third_party.py
```
## Real-ESRGAN Data Processing (for Training)
```bash
python src/third_party/Real-ESRGAN/scripts/generate_multiscale_DF2K.py \
    --input ./data/anime-images-raw \
    --output ./data/anime-images-multiscale
```
```bash
python src/third_party/Real-ESRGAN/scripts/generate_meta_info.py \
--input ./data/anime-images-raw ./data/anime-images-multiscale \
--root ./data ./data \
--meta_info "./data/meta_info/meta_info_multiscale.txt"
```
## Training
```bash
python src/anime_super_resolution/train.py \
    --config configs/Real-ESRGAN-Anime-finetuning.yml \
    --auto_resume
```

## Inference

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