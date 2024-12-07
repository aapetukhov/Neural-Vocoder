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

    audios = [audio for item in dataset_items for audio in item['audio']]
    audios = pad_sequence(audios, batch_first=True)

    spectrograms = [
        spectrogram.transpose(1, 2) for item in dataset_items for spectrogram in item['spectrogram']
    ]
    spectrograms = pad_sequence(spectrograms, batch_first=True, padding_value=PAD_CONST).transpose(1, 2)
    texts = sum((item['text'] for item in dataset_items), [])

    return {
        'audio': audios,
        'spectrogram': spectrograms,
        'text': texts
    }
