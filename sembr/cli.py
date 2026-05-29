import os
import sys
import traceback
from pathlib import Path


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def _safe_isatty(stream):
    try:
        return os.isatty(stream.fileno())
    except (AttributeError, OSError):
        return False


STDIN_TTY = _safe_isatty(sys.stdin)
STDOUT_TTY = _safe_isatty(sys.stdout)
STDERR_TTY = _safe_isatty(sys.stderr)

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


def package_version():
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version('sembr')
    except PackageNotFoundError:
        from . import __version__
        return __version__


def cli_parser():
    import argparse
    p = argparse.ArgumentParser(
        description='SemBr: Rewrap text with semantic breaks.')
    p.add_argument(
        '-V', '--version', action='version', version=package_version())
    p.add_argument(
        '-v', '--verbose', action='store_true', help='Enable verbose output')
    p.add_argument('-i', '--input-file', type=str, default=None)
    p.add_argument('-o', '--output-file', type=str, default=None)
    p.add_argument(
        '-c', '--config', action='append', default=[], metavar='KEY=VALUE',
        help='Override a config value from $XDG_CONFIG_HOME/sembr/config.toml')
    p.add_argument('-l', '--listen', action='store_true')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--mcp', action='store_true', help='Start MCP server mode')
    p.add_argument(
        '--file-type', type=str, default=None,
        help=(
            'File type (plaintext, latex, markdown, etc.). '
            'Auto-detect if not provided. '
            'File type must be provided if using stdin.'))
    return p


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


def apply_config(args, config):
    for key, value in config.items():
        setattr(args, key, value)
    return args


def init(model_name, bits=None, dtype=None, file_type=None, file_path=None, text=None, verbose=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from .processors import get_processor

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    dtype = getattr(torch, dtype) if dtype is not None else torch.float32
    kwargs = {}
    device = None
    if torch.cuda.is_available():
        if bits == 4:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
            kwargs['device_map'] = 'cuda'
        elif bits == 8:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_8bit=True)
            kwargs['device_map'] = 'cuda'
        else:
            device = 'cuda'
    elif torch.backends.mps.is_available():
        if bits in [4, 8]:
            raise RuntimeError('MPS does not support quantization.')
        device = 'mps'

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, torch_dtype=dtype, **kwargs)
    except Exception:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, torch_dtype=dtype, local_files_only=True, **kwargs)

    if device is not None:
        model = model.to(device)
    model.eval()
    processor = get_processor(
        file_type=file_type, file_path=file_path, text=text, verbose=verbose)
    return tokenizer, model, processor


def start_server(
    port, tokenizer, model, default_file_type=None, wrap_kwargs=None
):
    from flask import Flask, request
    from .processors import get_processor
    app = Flask(__name__)
    base_rv = {
        'model': model.__class__.__name__,
        'tokenizer': tokenizer.__class__.__name__,
    }

    @app.route('/check')
    def check():
        return {
            'status': 'success',
            **base_rv,
        }

    @app.route('/rewrap', methods=['POST'])
    def rewrap():
        from .inference import sembr
        form = request.form
        text = form['text']
        kwargs = dict(wrap_kwargs or {})

        # Get file_type from form data or use default
        file_type = form.get('file_type', default_file_type)

        # Create processor dynamically based on file type or text content
        processor = get_processor(
            file_type=file_type, text=text if not file_type else None)

        # Process other form parameters
        for k, v in form.items():
            if k in ['text', 'file_type']:
                continue
            if k in [
                'batch_size',
                'overlap_divisor',
                'min_tokens_per_line',
                'max_tokens_per_line',
            ]:
                v = int(v)
            if k in ['length_loss_weight']:
                v = float(v)
            kwargs[k] = v
        try:
            results = sembr(text, tokenizer, model, processor, **kwargs)
            return {
                'status': 'success',
                **base_rv,
                'processor': processor.__class__.__name__,
                'file_type': file_type,
                **kwargs,
                'text': results,
            }
        except Exception as e:
            return {
                'status': 'error',
                **base_rv,
                'processor': processor.__class__.__name__,
                'file_type': file_type,
                **kwargs,
                'error': str(e),
                'traceback': traceback.format_exc(),
            }

    app.run(port=port)


def _fetch(server, port, endpoint, method='get', data=None, timeout=None):
    import requests
    from requests.exceptions import ConnectionError, ReadTimeout
    try:
        results = getattr(requests, method.lower())(
            f'http://{server}:{port}/{endpoint}', data=data, timeout=timeout)
    except (ConnectionError, ReadTimeout) as e:
        raise RuntimeError(f'Connection Error: {e}')
    if results.status_code != 200:
        raise RuntimeError(
            f'Connection Error: {results.status_code}: {results.text}')
    data = results.json()
    if data['status'] != 'success':
        raise RuntimeError(
            f'Status: {data["status"]}\n'
            f'Exception: {data["error"]}\n'
            f'Traceback: {data.get("traceback")}')
    return data


def check_server(server, port):
    if not server:
        return False
    try:
        _fetch(server, port, 'check', timeout=0.3)
    except RuntimeError:
        return False
    return True


def rewrap_on_server(text, server, port, kwargs):
    data = {'text': text, **kwargs}
    response = _fetch(server, port, 'rewrap', 'post', data)
    return response['text']


def wrap_kwargs(args):
    return {
        'batch_size': args.batch_size,
        'predict_func': args.predict_func,
        'min_tokens_per_line': args.min_tokens_per_line,
        'max_tokens_per_line': args.max_tokens_per_line,
        'length_loss_weight': args.length_loss_weight,
        'overlap_divisor': args.overlap_divisor,
    }


def main() -> int:
    parser = cli_parser()
    args = parser.parse_args()
    try:
        apply_config(args, load_config(args.config))
    except Exception as e:
        print(f'Config error: {e}', file=sys.stderr)
        return 2
    if args.debug:
        import debugpy
        debugpy.listen(5678)
        print('Waiting for debugger to attach...')
        debugpy.wait_for_client()
    if args.mcp:
        from .mcp import mcp
        unsupported = ['input_file', 'output_file', 'listen']
        for arg_name in unsupported:
            if getattr(args, arg_name) in [None, False]:
                continue
            message = f'--{arg_name} is not supported in MCP mode.'
            print(message, file=sys.stderr)
            return 1
        mcp.run()
        return 0
    kwargs = wrap_kwargs(args)
    if args.listen:
        tokenizer, model, _ = init(
            args.model_name, args.bits, args.dtype, args.file_type, None, None, args.verbose)
        start_server(args.port, tokenizer, model, args.file_type, kwargs)
        return 0
    if args.input_file is not None:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif not STDIN_TTY:
        text = sys.stdin.read()
    else:
        parser.print_help()
        print('\nNo input file or stdin text provided.', file=sys.stderr)
        return 1
    if check_server(args.server, args.port):
        result = rewrap_on_server(text, args.server, args.port, kwargs)
    else:
        from .inference import sembr
        tokenizer, model, processor = init(
            args.model_name, args.bits, args.dtype,
            args.file_type, args.input_file, text, args.verbose)
        result = sembr(text, tokenizer, model, processor, **kwargs)
    if args.output_file is None:
        print(result)
        return 0
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    return 0


if __name__ == '__main__':
    sys.exit(main())
