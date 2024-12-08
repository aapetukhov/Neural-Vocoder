import torch
from torch.nn.utils.rnn import pad_sequence

PAD_CONST = -11.5129251

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    # not on device
    texts = [item['text'] for item in dataset_items]

    # on device
    audios = [
        audio
        for item in dataset_items
        for audio in (item['audio'] if item['audio'] is not None else [])
    ]
    audios = pad_sequence(audios, batch_first=True) if audios else audios

    spectrograms = [
        frame
        for item in dataset_items
        for frame in item['spectrogram'].transpose(1, 2)
    ]
    spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=PAD_CONST).transpose(1, 2)

    return {
        "audio": audios,
        "spectrogram": spectrograms,
        "text": texts
    }
