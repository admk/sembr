import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sembr.cli import MissingBackendError, init


class DummyModel:
    pass


class DummyTokenizer:
    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls()


def test_init_routes_mlx_to_loader(monkeypatch):
    captured = {}

    def fake_loader(model_name, *, dtype=None, quantization='none'):
        captured['model_name'] = model_name
        captured['dtype'] = dtype
        captured['quantization'] = quantization
        return DummyModel()

    expected_tokenizer = object()
    monkeypatch.setattr(
        'sembr.mlx.MlxTokenizer.from_pretrained',
        lambda model_name: expected_tokenizer)
    monkeypatch.setattr(
        'sembr.mlx.load_mlx_bert_token_classifier',
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


def test_init_prompts_for_mlx_extra_when_backend_dependency_missing(monkeypatch):
    def missing_tokenizer(model_name):
        raise ModuleNotFoundError("No module named 'tokenizers'")

    monkeypatch.setattr(
        'sembr.mlx.MlxTokenizer.from_pretrained',
        missing_tokenizer)

    try:
        init('local-model', file_type='plaintext', backend='mlx')
    except MissingBackendError as e:
        message = str(e)
        assert 'model.backend="mlx"' in message
        assert 'uv tool install "sembr[mlx]"' in message
    else:
        raise AssertionError('init accepted a missing MLX backend dependency')
