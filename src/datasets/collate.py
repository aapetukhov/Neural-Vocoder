from torch.nn.utils.rnn import pad_sequence
from itertools import chain

PAD_CONST = -11.51292

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
    msg = f"SPEC SHAPE IS: {dataset_items[0]['spectrogram'].shape}"
    print("-"*len(msg))
    print(msg)
    print("-"*len(msg))

    audios = [audio for item in dataset_items for audio in item["audio"]]
    audios = pad_sequence(audios, batch_first=True)

    spectrograms = list(chain.from_iterable(item['spectrogram'].transpose(1, 2) for item in dataset_items))
    texts = sum((item["text"] for item in dataset_items), [])

    return {
        "audio": audios,
        "spectrogram": spectrograms,
        "text": texts
    }
