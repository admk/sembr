import os
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError
from pydantic import field_validator, model_validator

from .platforms import (
    platform_override_keys,
    should_use_installed_cuda_extra,
    should_use_installed_mlx_backend,
    should_use_installed_torch_backend,
    warn_apple_silicon_torch_backend,
)


class SembrConfig(BaseModel):
    model_name: str = Field(alias='model.name')
    backend: Literal['torch', 'mlx', 'cuda'] = Field(alias='model.backend')
    bits: int | None = Field(default=None, alias='model.bits')
    dtype: str | None = Field(default=None, alias='model.dtype')
    quantization: Literal['none', 'affine', 'mxfp4', 'mxfp8', 'nvfp4'] = Field(
        alias='model.quantization')
    batch_size: PositiveInt = Field(alias='inference.batch_size')
    overlap_divisor: PositiveInt = Field(alias='inference.overlap_divisor')
    predict_func: Literal[
        'argmax',
        'logit_adjustment',
        'greedy_linebreaks',
        'balanced_linebreaks',
    ] = Field(alias='optimize.algorithm')
    preferred_min_tokens_per_line: PositiveInt | None = Field(
        default=None, alias='optimize.preferred_min_tokens_per_line')
    preferred_max_tokens_per_line: PositiveInt | None = Field(
        default=None, alias='optimize.preferred_max_tokens_per_line')
    line_length_penalty_weight: float = Field(
        ge=0, alias='optimize.line_length_penalty_weight')
    spaces: PositiveInt | Literal['auto'] = Field(
        alias='format.num_spaces')
    indent_type: Literal['space', 'tab', 'auto'] = Field(
        alias='format.indent_type')
    host: str = Field(alias='listen.host')
    port: PositiveInt = Field(alias='listen.port')

    model_config = ConfigDict(populate_by_name=True, extra='forbid')

    @field_validator('bits')
    @classmethod
    def _validate_bits(cls, value):
        if value is not None and value not in [4, 8]:
            raise ValueError('must be one of: 4, 8.')
        return value

    @model_validator(mode='after')
    def _validate_line_length_bounds(self):
        minimum = self.preferred_min_tokens_per_line
        maximum = self.preferred_max_tokens_per_line
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(
                'Config key "optimize.preferred_min_tokens_per_line" '
                'must be less than or equal to '
                '"optimize.preferred_max_tokens_per_line".')
        return self


def _valid_config_keys():
    return tuple(
        field.alias for field in SembrConfig.model_fields.values()
    )


def _format_error_location(location):
    if not location:
        return ''
    return '.'.join(str(part) for part in location)


def _clean_pydantic_message(message):
    prefix = 'Value error, '
    if message.startswith(prefix):
        return message[len(prefix):]
    return message


def _config_key_tokens(key):
    return frozenset(
        token for token in re.split(r'[._-]+', key) if token)


def _hamming_distance(left, right):
    left_tokens = _config_key_tokens(left)
    right_tokens = _config_key_tokens(right)
    vocabulary = left_tokens | right_tokens
    return sum(
        (token in left_tokens) != (token in right_tokens)
        for token in vocabulary)


def _normalized_hamming_distance(left, right):
    max_length = len(_config_key_tokens(left) | _config_key_tokens(right))
    if max_length == 0:
        return 0
    return _hamming_distance(left, right) / max_length


def _config_key_section(key):
    return key.split('.', 1)[0]


def _config_key_leaf(key):
    return key.rsplit('.', 1)[-1]


def _closest_config_key(key):
    valid_keys = _valid_config_keys()
    same_leaf = [
        valid_key
        for valid_key in valid_keys
        if _config_key_leaf(valid_key) == _config_key_leaf(key)
    ]
    if same_leaf:
        return min(
            same_leaf,
            key=lambda valid_key: (
                _normalized_hamming_distance(key, valid_key),
                _hamming_distance(key, valid_key),
                valid_key))
    same_section = [
        valid_key
        for valid_key in valid_keys
        if _config_key_section(valid_key) == _config_key_section(key)
    ]
    if same_section:
        valid_keys = same_section
    closest = min(
        valid_keys,
        key=lambda valid_key: (
            _normalized_hamming_distance(key, valid_key),
            _hamming_distance(key, valid_key),
            valid_key))
    if _normalized_hamming_distance(key, closest) <= 0.5:
        return closest
    return None


def _format_unknown_key_error(key):
    closest = _closest_config_key(key)
    if closest:
        return (
            f'Unknown config key "{key}"; '
            f'did you mean "{closest}"?')
    return (
        f'Unknown config key "{key}". '
        f'Valid keys are: {", ".join(_valid_config_keys())}.')


def _format_validation_error_item(error):
    key = _format_error_location(error.get('loc', ()))
    error_type = error.get('type')
    message = _clean_pydantic_message(error.get('msg', 'Invalid value'))
    value = error.get('input')
    if error_type == 'extra_forbidden':
        return _format_unknown_key_error(key)
    if not key:
        return message
    if error_type == 'literal_error':
        expected = error.get('ctx', {}).get('expected')
        if expected:
            return (
                f'Invalid value for "{key}": {value!r}. '
                f'Expected {expected}.')
    return f'Invalid value for "{key}": {value!r}. {message}.'


def _format_validation_error(error):
    details = [
        _format_validation_error_item(item)
        for item in error.errors(include_url=False)
    ]
    return 'Validation failed:\n' + '\n'.join(
        f'  - {detail}' for detail in details)


def _validate_config(data):
    try:
        return SembrConfig.model_validate(data)
    except ValidationError as e:
        raise ValueError(_format_validation_error(e)) from None


def config_path():
    return Path(
        os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')
    ) / 'sembr' / 'config.toml'


def default_config_path():
    return Path(__file__).with_name('default.toml')


def _is_default_config_path(path):
    return Path(path).resolve() == default_config_path().resolve()


def _load_toml(path):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with path.open('rb') as f:
        return tomllib.load(f)


def _flatten_section(section, values):
    config = {}
    section = section.replace('-', '_')
    if not isinstance(values, dict):
        raise ValueError(f'Config section [{section}] must be a TOML table.')
    for key, value in values.items():
        config_key = f'{section}.{key}'.replace('-', '_')
        config[config_key] = value
    return config


def _flatten_config_table(loaded):
    config = {}
    if not isinstance(loaded, dict):
        raise ValueError('Config file must contain a TOML table.')
    for section, values in loaded.items():
        if section == 'platform':
            continue
        config.update(_flatten_section(section, values))
    platform_tables = loaded.get('platform', {})
    if platform_tables:
        if not isinstance(platform_tables, dict):
            raise ValueError('Config section [platform] must be a TOML table.')
        for key in platform_override_keys():
            values = platform_tables.get(key, {})
            if values:
                config.update(_flatten_config_table(values))
    return config


def _parse_config_override(override):
    if '=' not in override:
        raise ValueError(
            f'Config override {override!r} must use KEY=VALUE syntax.')
    key, value = override.split('=', 1)
    key = key.strip().replace('-', '_')
    value = value.strip()
    if value.lower() in ['none', 'null'] and key != 'model.quantization':
        value = None
    return key, value


CONFIG_FILE_DEFAULTS = _flatten_config_table(_load_toml(default_config_path()))
CONFIG_DEFAULTS = SembrConfig.model_validate(CONFIG_FILE_DEFAULTS).model_dump()


def _config_data_from_attrs(config):
    return {
        field.alias: config[attr]
        for attr, field in SembrConfig.model_fields.items()
        if attr in config
    }


def _apply_installed_extra_backend(data, explicit_keys):
    if data.get('model.backend') != 'mlx':
        return data
    if should_use_installed_mlx_backend():
        return data
    if should_use_installed_cuda_extra():
        backend = 'cuda'
    elif should_use_installed_torch_backend():
        backend = 'torch'
    else:
        return data
    data = dict(data)
    data['model.backend'] = backend
    if (
        'model.name' not in explicit_keys
        or data.get('model.name') == 'admko/sembr2023-bert-small-nvfp4'
    ):
        data['model.name'] = 'admko/sembr2023-bert-small'
    if data.get('model.quantization') != 'none':
        data['model.quantization'] = 'none'
    warn_apple_silicon_torch_backend(backend)
    return data


def load_config(
    overrides=None, path=None, read_config_file=True, base_config=None
):
    data = dict(CONFIG_FILE_DEFAULTS)
    explicit_keys = set()
    if read_config_file:
        path = Path(path) if path is not None else config_path()
        if path.exists():
            file_config = _flatten_config_table(_load_toml(path))
            if not _is_default_config_path(path):
                explicit_keys.update(file_config)
            data.update(file_config)
    base_data = _config_data_from_attrs(base_config or {})
    explicit_keys.update(base_data)
    data.update(base_data)
    for override in overrides or []:
        key, value = _parse_config_override(override)
        explicit_keys.add(key)
        data[key] = value
    data = _apply_installed_extra_backend(data, explicit_keys)
    return _validate_config(data).model_dump()


def apply_config(
    args, overrides=None, path=None, read_config_file=True, base_config=None
):
    config = load_config(overrides, path, read_config_file, base_config)
    for key, value in config.items():
        setattr(args, key, value)
    return args
