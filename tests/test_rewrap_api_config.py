import os
import sys

from werkzeug.datastructures import MultiDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sembr.cli import rewrap_on_server, start_server


class DummyModel:
    pass


class DummyTokenizer:
    pass


def test_rewrap_accepts_repeated_cli_config_fields(monkeypatch):
    captured = {}

    def fake_sembr(text, tokenizer, model, processor, **kwargs):
        captured['text'] = text
        captured['processor'] = processor
        captured['kwargs'] = kwargs
        return 'wrapped'

    monkeypatch.setattr('sembr.inference.sembr', fake_sembr)
    monkeypatch.setattr('flask.Flask.run', lambda self, port: None)
    app = start_server(
        8384,
        DummyTokenizer(),
        DummyModel(),
        default_file_type='plaintext',
        default_config={
            'batch_size': 8,
            'predict_func': 'argmax',
            'preferred_min_tokens_per_line': None,
            'preferred_max_tokens_per_line': None,
            'line_length_penalty_weight': 0.05,
            'overlap_divisor': 8,
            'spaces': 4,
            'indent_type': 'space',
        },
    )

    response = app.test_client().post('/rewrap', data=MultiDict([
        ('text', 'hello world'),
        ('config', 'inference.batch-size=16'),
        ('config', 'optimize.algorithm=balanced_linebreaks'),
        ('config', 'optimize.preferred_min_tokens_per_line=8'),
        ('config', 'optimize.preferred_max_tokens_per_line=12'),
        ('config', 'format.num_spaces=2'),
        ('config', 'format.indent_type=tab'),
    ]))

    data = response.get_json()
    assert data['status'] == 'success'
    assert data['text'] == 'wrapped'
    assert captured['text'] == 'hello world'
    assert captured['kwargs']['batch_size'] == 16
    assert captured['kwargs']['predict_func'] == 'balanced_linebreaks'
    assert captured['kwargs']['preferred_min_tokens_per_line'] == 8
    assert captured['kwargs']['preferred_max_tokens_per_line'] == 12
    assert captured['processor'].__class__.__name__ == 'PlainTextProcessor'


def test_rewrap_on_server_sends_config_key_value_pairs(monkeypatch):
    captured = {}

    def fake_fetch(server, port, endpoint, method='get', data=None, timeout=None):
        captured['server'] = server
        captured['port'] = port
        captured['endpoint'] = endpoint
        captured['method'] = method
        captured['data'] = data
        return {'text': 'wrapped'}

    monkeypatch.setattr('sembr.cli._fetch', fake_fetch)

    result = rewrap_on_server(
        'hello',
        '127.0.0.1',
        8384,
        {
            'batch_size': 16,
            'preferred_max_tokens_per_line': None,
            'spaces': 'auto',
        },
        file_type='markdown',
    )

    assert result == 'wrapped'
    assert captured['endpoint'] == 'rewrap'
    assert captured['method'] == 'post'
    assert captured['data'] == [
        ('text', 'hello'),
        ('file_type', 'markdown'),
        ('config', 'inference.batch_size=16'),
        ('config', 'optimize.preferred_max_tokens_per_line=null'),
        ('config', 'format.num_spaces=auto'),
    ]
