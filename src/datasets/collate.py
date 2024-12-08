from torch.nn.utils.rnn import pad_sequence

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

    spectrograms = []
    for item in dataset_items:
        spectrogram = item["spectrogram"]
        if spectrogram.ndim == 3:
            spectrogram = spectrogram.squeeze(0)
        spectrograms.append(spectrogram.transpose(1, 2))
    spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=PAD_CONST).transpose(1, 2)
    texts = sum((item["text"] for item in dataset_items), [])

    return {
        "audio": audios,
        "spectrogram": spectrograms,
        "text": texts
    }
