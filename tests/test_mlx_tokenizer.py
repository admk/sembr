import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sembr.tokenizers import MlxTokenizer, NumpyTokenClassificationCollator


MODEL_PATH = 'checkpoints/sembr-bert-small-nvfp4'


def test_mlx_tokenizer_returns_ids_and_offsets():
    tokenizer = MlxTokenizer.from_pretrained(MODEL_PATH)

    encoding = tokenizer('Hello world', return_offsets_mapping=True)

    assert len(encoding.input_ids) == len(encoding.offset_mapping)
    assert encoding.input_ids[0] == 101
    assert encoding.offset_mapping[1] == (0, 5)


def test_mlx_tokenizer_adds_replacement_tokens():
    tokenizer = MlxTokenizer.from_pretrained(MODEL_PATH)

    added = tokenizer.add_tokens(['<SEMBr_TEST_TOKEN>'])
    encoding = tokenizer('<SEMBr_TEST_TOKEN>', return_offsets_mapping=True)

    assert added == 1
    assert len(encoding.input_ids) == 3


def test_numpy_token_classification_collator_pads_batches():
    tokenizer = MlxTokenizer.from_pretrained(MODEL_PATH)
    collator = NumpyTokenClassificationCollator(tokenizer)

    batch = collator([
        {'input_ids': [1, 2, 3]},
        {'input_ids': [4]},
    ])

    assert isinstance(batch['input_ids'], np.ndarray)
    assert batch['input_ids'].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch['attention_mask'].tolist() == [[1, 1, 1], [1, 0, 0]]
