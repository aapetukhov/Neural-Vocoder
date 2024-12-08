import json
import os
import shutil
import random
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

import logging

from src.datasets.base_dataset import BaseDataset

ROOT_PATH = Path("/kaggle/input/the-lj-speech-dataset/LJSpeech-1.1")
WORKING_PATH = Path("/kaggle/working/Neural-Vocoder")

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
}


logging.getLogger('speechbrain').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class LJspeechDatasetKaggle(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        index_path = WORKING_PATH / "data_index"
        index_path.mkdir(exist_ok=True, parents=True)
        
        if data_dir is None:
            data_dir = ROOT_PATH
        self._data_dir = data_dir
        self._index_dir = index_path
        
        index = self._get_or_load_index(part)
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._index_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        wav_dir = self._data_dir / "wavs"
        trans_path = self._data_dir / "metadata.csv"

        if not wav_dir.exists() or not trans_path.exists():
            raise FileNotFoundError(f"Missing required dataset files: {wav_dir} or {trans_path}.")

        files = list(wav_dir.glob("*.wav"))
        if len(files) == 0:
            raise ValueError("No .wav files found in the dataset.")

        metadata = {}
        with trans_path.open() as f:
            for line in f:
                w_id, _, w_text = line.strip().split("|")
                metadata[w_id] = w_text

        train_length = int(0.85 * len(files))
        if part == "train":
            files = files[:train_length]
        elif part == "test":
            files = files[train_length:]
        else:
            raise ValueError(f"Unknown dataset part: {part}")

        for wav_file in tqdm(files, desc=f"Preparing index for {part}"):
            w_id = wav_file.stem
            if w_id not in metadata:
                continue
            t_info = torchaudio.info(str(wav_file))
            length = t_info.num_frames / t_info.sample_rate
            index.append({
                "path": str(wav_file.absolute()),
                "text": metadata[w_id].lower(),
                "audio_len": length,
            })

        return index
