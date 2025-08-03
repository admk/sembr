import os
import sys
import traceback


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

STDIN_TTY = os.isatty(sys.stdin.fileno())
STDOUT_TTY = os.isatty(sys.stdout.fileno())
STDERR_TTY = os.isatty(sys.stderr.fileno())


def cli_parser():
    import argparse
    from . import __version__
    from .inference import PREDICT_FUNC_MAP
    p = argparse.ArgumentParser(
        description='SemBr: Rewrap text with semantic breaks.')
    model_name = 'admko/sembr2023-bert-small'
    p.add_argument('-v', '--version', action='version', version=__version__)
    p.add_argument('-m', '--model-name', type=str, default=model_name)
    p.add_argument('-i', '--input-file', type=str, default=None)
    p.add_argument('-o', '--output-file', type=str, default=None)
    p.add_argument('-b', '--batch-size', type=int, default=8)
    p.add_argument('-d', '--overlap-divisor', type=int, default=8)
    p.add_argument(
        '-f', '--predict-func', type=str,
        choices=PREDICT_FUNC_MAP, default='argmax')
    p.add_argument('-t', '--tokens-per-line', type=int, default=None)
    p.add_argument('-s', '--server', type=str, default='127.0.0.1')
    p.add_argument('-l', '--listen', action='store_true')
    p.add_argument('-p', '--port', type=int, default=8384)
    p.add_argument('--bits', type=int, choices=[4, 8], default=None)
    p.add_argument('--dtype', type=str, default=None)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--mcp', action='store_true', help='Start MCP server mode')
    p.add_argument(
        '--file-type', type=str, default=None,
        help=(
            'File type (plaintext, latex, markdown, etc.). '
            'Auto-detect if not provided. '
            'File type must be provided if using stdin.'))
    return p


def init(model_name, bits=None, dtype=None, file_type=None, file_path=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from .processors import get_processor
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = getattr(torch, dtype) if dtype is not None else torch.float32
    kwargs = {}
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        if bits == 4:
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        elif bits == 8:
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_8bit=True)
        kwargs['device_map'] = 'cuda'
    elif torch.backends.mps.is_available():
        if bits in [4, 8]:
            raise RuntimeError('MPS does not support quantization.')
        kwargs['device_map'] = 'mps'
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, torch_dtype=dtype, **kwargs)
    model.eval()
    processor = get_processor(file_type=file_type, file_path=file_path)
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
            if k in ['batch_size', 'tokens_per_line', 'overlap_divisor']:
                v = int(v)
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
        'tokens_per_line': args.tokens_per_line,
        'overlap_divisor': args.overlap_divisor,
    }


def main() -> int:
    parser = cli_parser()
    args = parser.parse_args()
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
            args.model_name, args.bits, args.dtype, args.file_type)
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
            args.file_type, args.input_file)
        result = sembr(text, tokenizer, model, processor, **kwargs)
    if args.output_file is None:
        print(result)
        return 0
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    return 0


if __name__ == '__main__':
    sys.exit(main())
