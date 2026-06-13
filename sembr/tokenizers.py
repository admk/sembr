from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Encoding:
    input_ids: list[int]
    offset_mapping: list[tuple[int, int]]


class SimpleBatch:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def to(self, device):
        return self


class MlxTokenizer:
    def __init__(self, tokenizer, pad_token_id=0):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id

    @classmethod
    def from_pretrained(cls, model_name):
        from tokenizers import Tokenizer

        path = _resolve_tokenizer_file(model_name)
        tokenizer = Tokenizer.from_file(str(path))
        pad_token_id = tokenizer.token_to_id('[PAD]')
        if pad_token_id is None:
            pad_token_id = 0
        return cls(tokenizer, pad_token_id)

    def add_tokens(self, tokens):
        return self.tokenizer.add_tokens(tokens)

    def __call__(self, text, return_offsets_mapping=False):
        enc = self.tokenizer.encode(text)
        offsets = list(enc.offsets) if return_offsets_mapping else []
        return Encoding(list(enc.ids), offsets)


class NumpyTokenClassificationCollator:
    def __init__(self, tokenizer, padding='longest'):
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, features, return_tensors=None):
        max_length = max(len(f['input_ids']) for f in features)
        input_ids, attention_mask = [], []
        for feature in features:
            ids = list(feature['input_ids'])
            pad_length = max_length - len(ids)
            input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_length)
            attention_mask.append([1] * len(ids) + [0] * pad_length)
        return SimpleBatch({
            'input_ids': np.asarray(input_ids, dtype=np.int64),
            'attention_mask': np.asarray(attention_mask, dtype=np.int64),
        })


def _resolve_tokenizer_file(model_name):
    path = Path(model_name)
    if path.exists():
        candidate = path / 'tokenizer.json' if path.is_dir() else path
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f'Could not find tokenizer.json in {path}.')

    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(model_name, 'tokenizer.json'))
