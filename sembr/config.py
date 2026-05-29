import os
from pathlib import Path


CONFIG_KEY_MAP = {
    'model.name': 'model_name',
    'model.bits': 'bits',
    'model.dtype': 'dtype',
    'inference.batch_size': 'batch_size',
    'inference.overlap_divisor': 'overlap_divisor',
    'optimize.algorithm': 'predict_func',
    'optimize.min_tokens_per_line': 'min_tokens_per_line',
    'optimize.max_tokens_per_line': 'max_tokens_per_line',
    'optimize.length_loss_weight': 'length_loss_weight',
    'server.ip': 'server',
    'server.port': 'port',
}

CONFIG_TYPES = {
    'model_name': str,
    'batch_size': int,
    'overlap_divisor': int,
    'predict_func': str,
    'min_tokens_per_line': int,
    'max_tokens_per_line': int,
    'length_loss_weight': float,
    'server': str,
    'port': int,
    'bits': int,
    'dtype': str,
}

CONFIG_NULLABLE = {
    'min_tokens_per_line',
    'max_tokens_per_line',
    'bits',
    'dtype',
}

PREDICT_FUNCS = (
    'argmax',
    'logit_adjustment',
    'greedy_linebreaks',
    'balanced_linebreaks',
)


def config_path():
    config_home = os.environ.get('XDG_CONFIG_HOME')
    if config_home is None:
        config_home = os.path.join(Path.home(), '.config')
    return Path(config_home) / 'sembr' / 'config.toml'


def default_config_path():
    return Path(__file__).with_name('default.toml')


def _load_toml(path):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with path.open('rb') as f:
        return tomllib.load(f)


def _config_key_to_attr(key):
    key = key.strip().replace('-', '_')
    if key not in CONFIG_KEY_MAP:
        valid = ', '.join(sorted(CONFIG_KEY_MAP))
        raise ValueError(f'Unknown config key {key!r}. Valid keys: {valid}.')
    return CONFIG_KEY_MAP[key]


def _flatten_config_table(loaded):
    valid_sections = sorted({key.split('.', 1)[0] for key in CONFIG_KEY_MAP})
    config = {}
    if not isinstance(loaded, dict):
        raise ValueError('Config file must contain a TOML table.')
    for section, values in loaded.items():
        section = section.replace('-', '_')
        if section not in valid_sections:
            valid = ', '.join(valid_sections)
            raise ValueError(
                f'Unknown config section [{section}]. Valid sections: {valid}.')
        if not isinstance(values, dict):
            raise ValueError(f'Config section [{section}] must be a TOML table.')
        for key, value in values.items():
            config_key = f'{section}.{key}'.replace('-', '_')
            attr = _config_key_to_attr(config_key)
            config[attr] = _parse_config_value(config_key, value)
    return config


def _parse_config_value(key, value):
    attr = _config_key_to_attr(key)
    if value is None:
        if attr in CONFIG_NULLABLE:
            return None
        raise ValueError(f'Config key {key!r} cannot be null.')
    expected_type = CONFIG_TYPES[attr]
    if expected_type in [int, float] and isinstance(value, bool):
        raise ValueError(
            f'Config key {key!r} must be {expected_type.__name__}.')
    if isinstance(value, str) and expected_type is not str:
        if value.lower() in ['none', 'null']:
            return _parse_config_value(key, None)
        try:
            value = expected_type(value)
        except ValueError:
            raise ValueError(
                f'Config key {key!r} must be {expected_type.__name__}.')
    if expected_type is float and isinstance(value, int):
        value = float(value)
    if not isinstance(value, expected_type):
        raise ValueError(
            f'Config key {key!r} must be {expected_type.__name__}.')
    if attr == 'predict_func' and value not in PREDICT_FUNCS:
        valid = ', '.join(PREDICT_FUNCS)
        raise ValueError(
            f'Config key {key!r} must be one of: {valid}.')
    if attr == 'bits' and value not in [4, 8]:
        raise ValueError(f"Config key {key!r} must be one of: 4, 8.")
    if attr in [
        'min_tokens_per_line',
        'max_tokens_per_line',
    ] and value < 1:
        raise ValueError(
            f'Config key {key!r} must be positive.')
    if attr == 'length_loss_weight' and value < 0:
        raise ValueError(
            f'Config key {key!r} must be non-negative.')
    return value


def _load_default_config():
    config = {key: None for key in CONFIG_NULLABLE}
    config.update(_flatten_config_table(_load_toml(default_config_path())))
    missing = sorted(set(CONFIG_TYPES) - set(config))
    if missing:
        raise RuntimeError(
            f'Default config is missing required keys: {", ".join(missing)}.')
    return config


CONFIG_DEFAULTS = _load_default_config()


def _parse_config_override(override):
    if '=' not in override:
        raise ValueError(
            f'Config override {override!r} must use KEY=VALUE syntax.')
    key, value = override.split('=', 1)
    key = key.strip().replace('-', '_')
    return _config_key_to_attr(key), _parse_config_value(key, value.strip())


def load_config(overrides=None, path=None):
    config = dict(CONFIG_DEFAULTS)
    path = Path(path) if path is not None else config_path()
    if path.exists():
        config.update(_flatten_config_table(_load_toml(path)))
    for override in overrides or []:
        attr, value = _parse_config_override(override)
        config[attr] = value
    _validate_config(config)
    return config


def _validate_config(config):
    minimum = config.get('min_tokens_per_line')
    maximum = config.get('max_tokens_per_line')
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(
            'Config key "optimize.min_tokens_per_line" must be less than or '
            'equal to "optimize.max_tokens_per_line".')


def apply_config(args, overrides=None, path=None):
    config = load_config(overrides, path)
    for key, value in config.items():
        setattr(args, key, value)
    return args
