import json
import logging
import shutil
from curses.ascii import isascii
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)


class LJspeechDatasetDatasphere(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = Path("/home/jupyter/work/resources/Neural-Vocoder")
        self._data_dir = data_dir
        index = self._get_or_load_index(part)
        self.index = index
        
        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        print("Loading LJSpeech")
        shutil.unpack_archive(arch_path, self._data_dir)
        
        dataset_dir = self._data_dir / "LJSpeech-1.1"
        wav_dir = dataset_dir / "wavs"
        metadata_path = dataset_dir / "metadata.csv"

        if not wav_dir.exists() or not metadata_path.exists():
            raise FileNotFoundError("Not found directory 'wavs' or 'metadata.csv' in archive.")
        
        return wav_dir, metadata_path

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        wav_dir, metadata_path = self._load_dataset()
        
        files = sorted(list(wav_dir.glob("*.wav")))
        train_length = int(0.85 * len(files))
        train_files = files[:train_length]
        test_files = files[train_length:]

        if part == "train":
            target_files = train_files
        elif part == "test":
            target_files = test_files
        else:
            raise ValueError(f"Invalid dataset part: {part}")

        index = []
        with metadata_path.open() as f:
            metadata = {line.split("|")[0]: " ".join(line.split("|")[1:]).strip() for line in f}
        
        for wav_file in tqdm(target_files, desc=f"Indexing {part} data"):
            w_id = wav_file.stem
            if w_id not in metadata:
                continue
            text = metadata[w_id].lower()
            audio_info = torchaudio.info(str(wav_file))
            audio_len = audio_info.num_frames / audio_info.sample_rate
            index.append({
                "path": str(wav_file.absolute().resolve()),
                "text": text,
                "audio_len": audio_len,
            })

        return index