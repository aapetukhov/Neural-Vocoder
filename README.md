# Neural Vocoder (HiFi-GAN) with PyTorch

Ссылка на отчёт: https://wandb.ai/aapetukhov-new-economic-school/Neural-Vocoder/reports/Neural-Vocoder-Updated---VmlldzoxMDY2ODUzOQ

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a project on Neural Vocoder HiFi-GAN with all necessary scripts provided for training and evaluating the model. It is worth noting that with better GPUs than a single P100 more extended training time would have been available, so higher results would have been achieved.

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## Hear with your ears
1. Inner-Analysis. Audios from train (1-st sample from LJSpeech). "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition." This is the same file as output_0 in input2speech_results, but in .mp4 format.
https://github.com/user-attachments/assets/2204a824-51da-43d9-bee3-7e0c556a57d5

## How To Train

To train a model, log in to wandb, clone the repo on Kaggle into the working area and run the following command:

```bash
python train.py -cn=kaggle_big_grad
```

Where all configs are from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

# How To Evaluate

Download the pretrained model from [here](https://drive.google.com/drive/folders/1pMJ1x6gwxQyf-twIPXTR-nAGzbraPduN?usp=sharing) and locate them in your directory.

1. To download from terminal:

```bash
# install gdown
pip install gdown

# download my model
gdown 1UH1s6hQSKoNIrokyk1_ZJc6Ud8b2HAaa
```

To run inference **LOCALLY** your dataset should have a strict structure. See, for example, `test_dataset` folder.

1. To generate an **audio which speaks arbitrary text** `<TEXT>` (which you print in the command line), run the command below. In my case, for example, `<MODEL_PATH>` is `"checkpoint-epoch20.pth"`.

```bash
python synthesize.py -cn=infer_input2speech \
   '+datasets.test.index=[{text: "<TEXT>", audio_len: 0, path: "anything.txt"}]' \
   'inferencer.from_pretrained="<MODEL_PATH>"'
```

2. To generate **audio from a given audio**, put your audios into `<AUDIO_DIR>` and their transcriptions to `<TRANS_DIR>` (see, for example, my `text_dataset/audios` and `text_dataset/transcriptions`).

```bash
python synthesize.py -cn=infer_speech2speech \
   '++datasets.test.audio_dir="<AUDIO_DIR>"' \
   '+datasets.test.transcription_dir="<TRANS_DIR>"' \
   'inferencer.from_pretrained="<MODEL_PATH>"'
```

3. To generate **audio from given texts in a form of dataset**, put your texts into `<TEXT_DIR>` (see, for example, my `text_dataset/transcriptions`) and run:

```bash
python synthesize.py -cn=infer_text2speech \
   '++datasets.test.data_dir="<TEXT_DIR>"' \
   'inferencer.from_pretrained="<MODEL_PATH>"'
```

After generation check the folder `saved_audios`, you will find your audios there

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
