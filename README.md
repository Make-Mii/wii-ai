# wii-ai

<div align="center">
  <img src="https://raw.githubusercontent.com/Make-Mii/wii-ai/main/assets/logo.png" alt="Make Mii Logo" width="300" />

  <p><strong>Real-time facial capture to playable Nintendo Wii Mii avatars</strong></p>

  <p>
    <a href="https://hacknotts.com">Built for HackNotts '26</a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/🏆_Best_AI%2FML_Project-HackNotts_'26-gold?style=for-the-badge" alt="Best AI/ML Project" />
    <img src="https://img.shields.io/badge/🥈_Runner--up-Telluvian_Best_Use_of_Multiple_AI_Systems-silver?style=for-the-badge" alt="Telluvian Runner-up" />
  </p>
</div>

## Overview

A real time computer vision and machine learning pipeline developed for ![HackNotts26](https://hacknotts.com/), the backend processing for [wii-make-mii](https://github.com/Make-Mii/wii-make-mii).
This system captures a user's face via a webcam, extracts facial feature landmarks using Google's MediaPipe Face Landmarker, hair segmentation masks using MediaPipe Hair Image Segmention and then matches the geometry and colours to Nintendo Mii assets using OpenAI's CLIP. That resulting Mii configuration is then transmitted over UDP directly to a Nintendo [Wii homebrew application](https://github.com/Make-Mii/wii-make-mii) which allows display and saving to the devices memory.

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/Make-Mii/wii-ai.git
cd wii-ai
```

1. Create and activate conda environment:

```bash
conda create -n wii-ai python=3.10 -y
conda activate wii-ai
```

1. Install dependencies:

```bash
pip install -r requirements.txt
cd src
```

1. (Optional) Update sprites:

```bash
python crop_sprites.py
python normalise_eye_eyebrows.py
python index_mii_components.py
```

1. Configure Wii IP

- Run the main application on the Wii, from there take the Wii's IP address from the top of the UI and update WII_IP at the top of `main.py`.

1. Run program

```bash
python main.py
```
