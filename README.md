# wii-ai

<div align="center">
  <img src="assets/logo.png" alt="Make Mii Logo" width="300" />

  <p><strong>Real-time Mii avatar generation and rendering for the Nintendo Wii homebrew ecosystem</strong></p>

  <p>
    <a href="https://hacknotts.com">Built for HackNotts '26</a>
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

2. Create and activate conda environment:

```bash
conda create -n wii-ai python=3.10 -y
conda activate wii-ai
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) Update sprites:

```bash
python src/crop_sprites.py
python normalise_eye_eyebrows.py
python index_mii_components.py
```

5. Configure Wii IP

- Run the main application on the Wii, from there take the Wii's IP address from the top of the UI and update WII_IP at the top of `main.py`.

6. Run program

```bash
python src/main.py
```

## Controls

Sliders Adjust:

- Brightness
- Contrast
- Gamma
- Saturation
in the `stream` window to ensure clean skin and hair feature separation.

