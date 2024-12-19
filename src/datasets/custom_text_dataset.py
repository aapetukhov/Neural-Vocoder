from pathlib import Path
from src.datasets.base_dataset import BaseDataset


class CustomDirTextDataset(BaseDataset):
    def __init__(self, data_dir: str = None, text: str = None, *args, **kwargs):
        if text:
            data = [{"text": text, "audio_len": 0}]
        elif data_dir:
            transcription_path = Path(data_dir)
            data = [
                { 
                    "audio_len": 0,
                    "path": str(path),
                    "text": path.read_text(encoding="utf-8").strip()
                    }
                for path in transcription_path.iterdir()
                if path.suffix.lower() == ".txt" and path.read_text(encoding="utf-8").strip()
            ]
        else:
            data = []

        super().__init__(data, *args, **kwargs)
