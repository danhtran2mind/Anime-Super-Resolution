# Anime Super Resolution 🖼️




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



```bash
git clone https://github.com/danhtran2mind/Anime-Super-Resolution
cd Anime-Super-Resolution
```
!pip install -r requirements/requirements.txt

!python scripts/download_ckpts.py

!python scripts/setup_third_party.py

```
python src/anime_super_resolution/train.py
```
```
python src/anime_super_resolution/infer.py
```