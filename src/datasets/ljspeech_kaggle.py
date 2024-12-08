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
WORKING_PATH = Path("/kaggle/working/Neural-Vocoder/LJSpeech")

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
}


logging.getLogger('speechbrain').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class LJspeechDatasetKaggle(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        index_path = Path("/kaggle/working/Neural-Vocoder/data_index")
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
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._prepare_splits()

        wav_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                wav_dirs.add(dirpath)
        
        for wav_dir in tqdm(list(wav_dirs), desc=f"Preparing LJSpeech folders: {part}"):
            wav_dir = Path(wav_dir)
            trans_path = self._data_dir / "metadata.csv"
            with trans_path.open() as f:
                for line in f:
                    w_id, w_text = line.split("|")[0], line.split("|")[2].strip()
                    wav_path = wav_dir / f"{w_id}.wav"
                    if not wav_path.exists(): 
                        continue
                    t_info = torchaudio.info(str(wav_path))
                    length = t_info.num_frames / t_info.sample_rate
                    if w_text.isascii():
                        index.append(
                            {
                                "path": str(wav_path.absolute().resolve()),
                                "text": w_text.lower(),
                                "audio_len": length,
                            }
                        )
        return index

    def _prepare_splits(self):
        print("Splitting dataset...")
        wav_dir = self._data_dir / "wavs"
        if not wav_dir.exists():
            raise FileNotFoundError(f"Expected wavs directory {wav_dir} not found.")

        files = list(wav_dir.iterdir())
        train_length = int(0.85 * len(files)) 
        train_dir = self._data_dir / "train"
        test_dir = self._data_dir / "test"
        train_dir.mkdir(exist_ok=True, parents=True)
        test_dir.mkdir(exist_ok=True, parents=True)
        
        for i, fpath in enumerate(files):
            if i < train_length:
                shutil.move(str(fpath), str(train_dir / fpath.name))
            else:
                shutil.move(str(fpath), str(test_dir / fpath.name))
