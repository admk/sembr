import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sembr.cli import init


class DummyModel:
    pass


class DummyTokenizer:
    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls()


def test_init_routes_mlx_backend_to_mlx_loader(monkeypatch):
    captured = {}

    def fake_loader(model_name, *, dtype=None, quantization='none'):
        captured['model_name'] = model_name
        captured['dtype'] = dtype
        captured['quantization'] = quantization
        return DummyModel()

    expected_tokenizer = object()
    monkeypatch.setattr(
        'sembr.cli._from_pretrained', lambda cls, name: expected_tokenizer)
    monkeypatch.setattr(
        'sembr.mlx_backend.load_mlx_bert_token_classifier',
        fake_loader)

    tokenizer, model, processor = init(
        'local-model',
        dtype='float16',
        file_type='plaintext',
        backend='mlx',
        quantization='nvfp4',
    )

    assert tokenizer is expected_tokenizer
    assert isinstance(model, DummyModel)
    assert processor.__class__.__name__ == 'PlainTextProcessor'
    assert captured == {
        'model_name': 'local-model',
        'dtype': 'float16',
        'quantization': 'nvfp4',
    }
