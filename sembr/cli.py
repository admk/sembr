import os
import sys
import traceback

from .config import SembrConfig, apply_config, load_config


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def _safe_isatty(stream):
    try:
        return os.isatty(stream.fileno())
    except (AttributeError, OSError):
        return False


def package_version():
    from importlib.metadata import PackageNotFoundError, version
    try:
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


def _from_pretrained(model_class, model_name, **kwargs):
    try:
        return model_class.from_pretrained(model_name, **kwargs)
    except Exception:
        return model_class.from_pretrained(
            model_name, local_files_only=True, **kwargs)


def init(
    model_name, bits=None, dtype=None, file_type=None, file_path=None,
    text=None, verbose=False, spaces=4, indent_type='space'
):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from .processors import get_processor

    tokenizer = _from_pretrained(AutoTokenizer, model_name)
    requested_dtype = getattr(torch, dtype) if dtype is not None else None
    compute_dtype = (
        requested_dtype if requested_dtype is not None else torch.float32)
    model_kwargs = {}
    device = None
    if torch.cuda.is_available():
        if bits == 4:
            from transformers import BitsAndBytesConfig
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype)
            model_kwargs['device_map'] = 'cuda'
        elif bits == 8:
            from transformers import BitsAndBytesConfig
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_8bit=True)
            model_kwargs['device_map'] = 'cuda'
        else:
            device = 'cuda'
    elif torch.backends.mps.is_available():
        if bits in [4, 8]:
            raise RuntimeError('MPS does not support quantization.')
        device = 'mps'
    if requested_dtype is not None:
        model_kwargs['torch_dtype'] = requested_dtype

    model = _from_pretrained(
        AutoModelForTokenClassification,
        model_name,
        **model_kwargs)
    if device is not None:
        model = model.to(device)
    model.eval()
    processor = get_processor(
        file_type, file_path, text, verbose,
        spaces=spaces, indent_type=indent_type)
    return tokenizer, model, processor


def rewrap_text(
    text, tokenizer, model, config, processor=None, file_type=None,
    file_path=None, verbose=False,
):
    from .inference import sembr
    from .processors import get_processor

    if processor is None:
        processor = get_processor(
            file_type, file_path, text, verbose, **config)
    return (
        processor,
        sembr(text, tokenizer, model, processor, **config),
    )


def start_server(
    host, port, tokenizer, model, default_file_type=None, default_config=None
):
    from flask import Flask, request
    app = Flask(__name__)
    base_rv = {
        'model': model.__class__.__name__,
        'tokenizer': tokenizer.__class__.__name__,
    }

    @app.route('/check')
    def check():
        return {'status': 'success', **base_rv}

    @app.route('/rewrap', methods=['POST'])
    def rewrap():
        form = request.form
        file_type = form.get('file_type', default_file_type)
        try:
            text = form['text']
            config = load_config(
                form.getlist('config'),
                read_config_file=False,
                base_config=default_config)
            processor, results = rewrap_text(
                text, tokenizer, model, config, file_type=file_type)
            return {
                'status': 'success',
                **base_rv,
                'processor': processor.__class__.__name__,
                'file_type': file_type,
                'text': results,
            }
        except Exception as e:
            return {
                'status': 'error',
                **base_rv,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'file_type': file_type,
            }

    app.run(host=host, port=port)
    return app


def _fetch(host, port, endpoint, method='get', data=None, timeout=None):
    import requests
    from requests.exceptions import ConnectionError, ReadTimeout
    try:
        results = getattr(requests, method.lower())(
            f'http://{host}:{port}/{endpoint}', data=data, timeout=timeout)
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


def check_server(host, port):
    if not host:
        return False
    try:
        _fetch(host, port, 'check', timeout=0.3)
    except RuntimeError:
        return False
    return True


def rewrap_on_server(
    text, host, port, config, file_type=None
):
    data = [('text', text)]
    if file_type is not None:
        data.append(('file_type', file_type))
    data.extend(
        (
            'config',
            f'{field.alias}='
            f'{"null" if config[attr] is None else config[attr]}',
        )
        for attr, field in SembrConfig.model_fields.items()
        if attr in config)
    response = _fetch(host, port, 'rewrap', 'post', data)
    return response['text']


def print_args(args):
    from pprint import pprint
    print('Arguments:', file=sys.stderr)
    pprint(vars(args), stream=sys.stderr, sort_dicts=True)


def main() -> int:
    parser = cli_parser()
    args = parser.parse_args()
    try:
        config = vars(apply_config(args, args.config))
    except Exception as e:
        print(f'Config error: {e}', file=sys.stderr)
        return 2
    if args.verbose:
        print_args(args)
    if args.mcp:
        from .mcp import mcp
        unsupported = ['input_file', 'output_file', 'listen']
        for arg_name in unsupported:
            if getattr(args, arg_name) in [None, False]:
                continue
            print(
                f'--{arg_name} is not supported in MCP mode.', file=sys.stderr)
            return 1
        mcp.run()
        return 0
    if args.listen:
        tokenizer, model, _ = init(
            args.model_name, args.bits, args.dtype, args.file_type, None, None,
            args.verbose, config['spaces'], config['indent_type'])
        start_server(
            args.host, args.port, tokenizer, model, args.file_type, config)
        return 0
    if args.input_file is not None:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif not _safe_isatty(sys.stdin):
        text = sys.stdin.read()
    else:
        parser.print_help()
        print('\nNo input file or stdin text provided.', file=sys.stderr)
        return 1
    if check_server(args.host, args.port):
        result = rewrap_on_server(
            text, args.host, args.port, config, args.file_type)
    else:
        tokenizer, model, processor = init(
            args.model_name, args.bits, args.dtype,
            args.file_type, args.input_file, text, args.verbose,
            config['spaces'], config['indent_type'])
        _, result = rewrap_text(text, tokenizer, model, config, processor)
    if args.output_file is None:
        print(result)
        return 0
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    return 0


if __name__ == '__main__':
    sys.exit(main())
