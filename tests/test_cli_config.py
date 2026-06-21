import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from sembr.cli import cli_parser
from sembr.config import (
    CONFIG_DEFAULTS,
    config_path,
    default_config_path,
    load_config,
    _flatten_config_table,
)
from sembr.platforms import install_command_for_extra


@pytest.fixture(autouse=True)
def disable_installed_backend_detection(monkeypatch):
    monkeypatch.setattr(
        'sembr.config.should_use_installed_cuda_extra',
        lambda: False)
    monkeypatch.setattr(
        'sembr.config.should_use_installed_mlx_backend',
        lambda: False)
    monkeypatch.setattr(
        'sembr.config.should_use_installed_torch_backend',
        lambda: False)


def test_platform_macos_arm64_default_uses_mlx(monkeypatch):
    monkeypatch.setattr(
        'sembr.config.platform_override_keys',
        lambda: ('darwin', 'macos', 'darwin-arm64', 'macos-arm64'))
    monkeypatch.setattr(
        'sembr.config.should_use_installed_cuda_extra',
        lambda: False)
    monkeypatch.setattr(
        'sembr.config.should_use_installed_mlx_backend',
        lambda: True)
    monkeypatch.setattr(
        'sembr.config.should_use_installed_torch_backend',
        lambda: False)

    config = load_config(path=default_config_path(), read_config_file=True)

    assert config['backend'] == 'mlx'
    assert config['model_name'] == 'admko/sembr2023-bert-small-nvfp4'
    assert config['quantization'] == 'nvfp4'


def test_platform_macos_arm64_cuda_extra_uses_cuda_with_warning(monkeypatch):
    monkeypatch.setattr(
        'sembr.config.platform_override_keys',
        lambda: ('darwin', 'macos', 'darwin-arm64', 'macos-arm64'))
    monkeypatch.setattr(
        'sembr.config.should_use_installed_cuda_extra',
        lambda: True)
    monkeypatch.setattr(
        'sembr.config.should_use_installed_mlx_backend',
        lambda: False)
    monkeypatch.setattr(
        'sembr.config.should_use_installed_torch_backend',
        lambda: True)

    with pytest.warns(RuntimeWarning, match='MLX extra is recommended'):
        config = load_config(
            path=default_config_path(), read_config_file=True)

    assert config['backend'] == 'cuda'
    assert config['model_name'] == 'admko/sembr2023-bert-small'
    assert config['quantization'] == 'none'


def test_platform_macos_arm64_cpu_extra_uses_torch_with_warning(monkeypatch):
    monkeypatch.setattr(
        'sembr.config.platform_override_keys',
        lambda: ('darwin', 'macos', 'darwin-arm64', 'macos-arm64'))
    monkeypatch.setattr(
        'sembr.config.should_use_installed_cuda_extra',
        lambda: False)
    monkeypatch.setattr(
        'sembr.config.should_use_installed_mlx_backend',
        lambda: False)
    monkeypatch.setattr(
        'sembr.config.should_use_installed_torch_backend',
        lambda: True)

    with pytest.warns(RuntimeWarning, match='model.backend="torch"'):
        config = load_config(
            path=default_config_path(), read_config_file=True)

    assert config['backend'] == 'torch'
    assert config['model_name'] == 'admko/sembr2023-bert-small'
    assert config['quantization'] == 'none'


def test_platform_macos_arm64_preserves_explicit_mlx_when_mlx_is_installed(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(
        'sembr.config.platform_override_keys',
        lambda: ('darwin', 'macos', 'darwin-arm64', 'macos-arm64'))
    monkeypatch.setattr(
        'sembr.config.should_use_installed_cuda_extra',
        lambda: True)
    monkeypatch.setattr(
        'sembr.config.should_use_installed_mlx_backend',
        lambda: True)
    monkeypatch.setattr(
        'sembr.config.should_use_installed_torch_backend',
        lambda: True)
    path = tmp_path / 'config.toml'
    path.write_text('[model]\nbackend = "mlx"', encoding='utf-8')

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        config = load_config(path=path)

    assert config['backend'] == 'mlx'


def test_platform_macos_arm64_explicit_mlx_falls_back_when_mlx_is_missing(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(
        'sembr.config.platform_override_keys',
        lambda: ('darwin', 'macos', 'darwin-arm64', 'macos-arm64'))
    monkeypatch.setattr(
        'sembr.config.should_use_installed_cuda_extra',
        lambda: False)
    monkeypatch.setattr(
        'sembr.config.should_use_installed_mlx_backend',
        lambda: False)
    monkeypatch.setattr(
        'sembr.config.should_use_installed_torch_backend',
        lambda: True)
    path = tmp_path / 'config.toml'
    path.write_text(
        '\n'.join([
            '[model]',
            'name = "admko/sembr2023-bert-small-nvfp4"',
            'backend = "mlx"',
            'quantization = "nvfp4"',
            'dtype = "bfloat16"',
        ]),
        encoding='utf-8')

    with pytest.warns(RuntimeWarning, match='model.backend="torch"'):
        config = load_config(path=path)

    assert config['backend'] == 'torch'
    assert config['model_name'] == 'admko/sembr2023-bert-small'
    assert config['quantization'] == 'none'
    assert config['dtype'] == 'bfloat16'


def test_platform_override_precedence(monkeypatch):
    monkeypatch.setattr(
        'sembr.config.platform_override_keys',
        lambda: ('linux', 'linux-x86_64'))

    config = _flatten_config_table({
        'model': {
            'backend': 'torch',
            'name': 'base-model',
        },
        'platform': {
            'linux': {
                'model': {
                    'backend': 'mlx',
                    'name': 'linux-model',
                },
            },
            'linux-x86_64': {
                'model': {
                    'name': 'linux-x86-model',
                },
            },
        },
    })

    assert config['model.backend'] == 'mlx'
    assert config['model.name'] == 'linux-x86-model'


def test_install_command_formats_extra():
    assert install_command_for_extra('mlx') == 'uv tool install "sembr[mlx]"'


def test_config_path_uses_xdg_config_home(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))

    assert config_path() == tmp_path / 'sembr' / 'config.toml'


def test_load_config_reads_toml_and_applies_overrides(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(
        '\n'.join([
            '[model]',
            'name = "local-model"',
            'backend = "mlx"',
            'bits = 8',
            'dtype = "float16"',
            'quantization = "nvfp4"',
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
            '[listen]',
            'host = "0.0.0.0"',
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
        'backend': 'mlx',
        'batch_size': 16,
        'overlap_divisor': 2,
        'predict_func': 'greedy_linebreaks',
        'preferred_max_tokens_per_line': None,
        'spaces': 8,
        'indent_type': 'tab',
        'host': '0.0.0.0',
        'port': 9000,
        'bits': 4,
        'dtype': 'float16',
        'quantization': 'nvfp4',
    }


def test_load_config_rejects_invalid_model_backend(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text('[model]\nbackend = "coreml"', encoding='utf-8')

    try:
        load_config(path=path)
    except ValueError as e:
        assert 'model.backend' in str(e)
        assert 'torch' in str(e)
        assert 'mlx' in str(e)
    else:
        raise AssertionError('load_config accepted an invalid backend')


def test_load_config_accepts_mlx_nvfp4_quantization(tmp_path):
    config = load_config(
        [
            'model.backend=mlx',
            'model.quantization=nvfp4',
        ],
        path=tmp_path / 'missing.toml')

    assert config['backend'] == 'mlx'
    assert config['quantization'] == 'nvfp4'


def test_load_config_accepts_quantization_none_override(tmp_path):
    config = load_config(
        ['model.quantization=none'],
        path=tmp_path / 'missing.toml')

    assert config['quantization'] == 'none'


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
        assert 'space' in str(e)
        assert 'tab' in str(e)
    else:
        raise AssertionError('load_config accepted an invalid indent type')


def test_load_config_accepts_auto_num_spaces(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text('[format]\nnum_spaces = "auto"', encoding='utf-8')

    config = load_config(path=path)

    assert config['spaces'] == 'auto'


def test_load_config_accepts_auto_indent_type(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text('[format]\nindent_type = "auto"', encoding='utf-8')

    config = load_config(path=path)

    assert config['indent_type'] == 'auto'


def test_load_config_rejects_unknown_key(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text('[model]\nunknown = "value"', encoding='utf-8')

    try:
        load_config(path=path)
    except ValueError as e:
        assert 'model.unknown' in str(e)
        assert 'Valid keys are:' in str(e)
        assert 'model.name' in str(e)
        assert 'Extra inputs are not permitted' not in str(e)
    else:
        raise AssertionError('load_config accepted an unknown key')


def test_load_config_rejects_legacy_option_names_with_replacements(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(
        '\n'.join([
            '[optimize]',
            'min_tokens_per_line = 8',
            'max_tokens_per_line = 12',
            'length_loss_weight = 0.05',
        ]),
        encoding='utf-8')

    try:
        load_config(path=path)
    except ValueError as e:
        message = str(e)
        assert 'optimize.min_tokens_per_line' in message
        assert 'optimize.preferred_min_tokens_per_line' in message
        assert 'optimize.max_tokens_per_line' in message
        assert 'optimize.preferred_max_tokens_per_line' in message
        assert 'optimize.length_loss_weight' in message
        assert 'optimize.line_length_penalty_weight' in message
        assert 'did you mean' in message
        assert 'use "' not in message
        assert 'Extra inputs are not permitted' not in message
        assert 'errors for SembrConfig' not in message
    else:
        raise AssertionError('load_config accepted legacy option names')


def test_load_config_rejects_unknown_key_with_suggestion(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(
        '[optimize]\npreferred_max_token_per_line = 12',
        encoding='utf-8')

    try:
        load_config(path=path)
    except ValueError as e:
        assert 'optimize.preferred_max_token_per_line' in str(e)
        assert 'did you mean "optimize.preferred_max_tokens_per_line"' in str(e)
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
        '--host',
        '--port',
        '--bits',
        '--dtype',
        '--backend',
        '--quantization',
    ]:
        assert option not in help_text
    assert '-c KEY=VALUE' in help_text
