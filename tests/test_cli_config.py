import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sembr.cli import cli_parser
from sembr.config import CONFIG_DEFAULTS, config_path, load_config


def test_config_path_uses_xdg_config_home(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))

    assert config_path() == tmp_path / 'sembr' / 'config.toml'


def test_load_config_reads_toml_and_applies_overrides(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(
        '\n'.join([
            '[model]',
            'name = "local-model"',
            'bits = 8',
            'dtype = "float16"',
            '',
            '[inference]',
            'batch_size = 4',
            'overlap_divisor = 2',
            '',
            '[optimize]',
            'algorithm = "greedy_linebreaks"',
            'preferred_max_tokens_per_line = 12',
            '',
            '[format]',
            'num_spaces = 2',
            'indent_type = "tab"',
            '',
            '[server]',
            'ip = "0.0.0.0"',
            'port = 9000',
        ]),
        encoding='utf-8')

    config = load_config(
        [
            'inference.batch-size=16',
            'optimize.preferred_max_tokens_per_line=null',
            'model.bits=4',
            'format.num_spaces=8',
        ],
        path=path)

    assert config == {
        **CONFIG_DEFAULTS,
        'model_name': 'local-model',
        'batch_size': 16,
        'overlap_divisor': 2,
        'predict_func': 'greedy_linebreaks',
        'preferred_max_tokens_per_line': None,
        'spaces': 8,
        'indent_type': 'tab',
        'server': '0.0.0.0',
        'port': 9000,
        'bits': 4,
        'dtype': 'float16',
    }


def test_load_config_accepts_balanced_linebreak_range(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(
        '\n'.join([
            '[optimize]',
            'algorithm = "balanced_linebreaks"',
            'preferred_min_tokens_per_line = 8',
            'preferred_max_tokens_per_line = 12',
            'line_length_penalty_weight = 0.05',
        ]),
        encoding='utf-8')

    config = load_config(path=path)

    assert config['predict_func'] == 'balanced_linebreaks'
    assert config['preferred_min_tokens_per_line'] == 8
    assert config['preferred_max_tokens_per_line'] == 12
    assert config['line_length_penalty_weight'] == 0.05


def test_load_config_accepts_balanced_linebreak_range_override(tmp_path):
    config = load_config(
        [
            'optimize.algorithm=balanced_linebreaks',
            'optimize.preferred_min_tokens_per_line=8',
            'optimize.preferred_max_tokens_per_line=12',
            'optimize.line_length_penalty_weight=0.05',
        ],
        path=tmp_path / 'missing.toml')

    assert config['predict_func'] == 'balanced_linebreaks'
    assert config['preferred_min_tokens_per_line'] == 8
    assert config['preferred_max_tokens_per_line'] == 12
    assert config['line_length_penalty_weight'] == 0.05


def test_load_config_rejects_invalid_line_length_bounds(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(
        '\n'.join([
            '[optimize]',
            'preferred_min_tokens_per_line = 12',
            'preferred_max_tokens_per_line = 8',
        ]),
        encoding='utf-8')

    try:
        load_config(path=path)
    except ValueError as e:
        assert 'optimize.preferred_min_tokens_per_line' in str(e)
    else:
        raise AssertionError('load_config accepted invalid line length bounds')


def test_load_config_rejects_invalid_indent_type(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text('[format]\nindent_type = "tabs"', encoding='utf-8')

    try:
        load_config(path=path)
    except ValueError as e:
        assert 'format.indent_type' in str(e)
        assert 'space, tab' in str(e)
    else:
        raise AssertionError('load_config accepted an invalid indent type')


def test_load_config_accepts_auto_num_spaces(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text('[format]\nnum_spaces = "auto"', encoding='utf-8')

    config = load_config(path=path)

    assert config['spaces'] == 'auto'


def test_load_config_rejects_unknown_key(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text('[model]\nunknown = "value"', encoding='utf-8')

    try:
        load_config(path=path)
    except ValueError as e:
        assert "Unknown config key 'model.unknown'" in str(e)
    else:
        raise AssertionError('load_config accepted an unknown key')


def test_removed_options_are_no_longer_cli_arguments():
    parser = cli_parser()
    help_text = parser.format_help()

    for option in [
        '--model-name',
        '--batch-size',
        '--overlap-divisor',
        '--predict-func',
        '--tokens-per-line',
        '--min-tokens-per-line',
        '--max-tokens-per-line',
        '--length-loss-weight',
        '--server',
        '--port',
        '--bits',
        '--dtype',
    ]:
        assert option not in help_text
    assert '-c KEY=VALUE' in help_text
