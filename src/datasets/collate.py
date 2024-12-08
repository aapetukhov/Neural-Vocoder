import torch
from torch.nn.utils.rnn import pad_sequence
from itertools import chain

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
    #not putting on device
    texts = [item['text'] for item in dataset_items]

    # on device
    audios = list(chain.from_iterable(item['audio'] if item['audio'] is not None else [] for item in dataset_items))
    audios = pad_sequence(audios, batch_first=True) if audios else audios

    spectrograms = list(chain.from_iterable(item['spectrogram'].transpose(1, 2) for item in dataset_items))
    spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=PAD_CONST).transpose(1, 2)

    return {
        "audio": audios,
        "spectrogram": spectrograms,
        "text": texts
    }
